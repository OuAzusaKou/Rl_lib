from gym.spaces import Discrete, Box
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from scipy.optimize import linprog
from pettingzoo.test import api_test
from pettingzoo.utils import random_demo


def wrapped_env(elec):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = MA_ELEC_ENV(elec)
    #env = wrappers.CaptureStdoutWrapper(env)
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class MA_ELEC_ENV(AECEnv):
    """
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, elec):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        # electronic need
        self.electronic_need = [elec]*10
        # episode length
        self.max_step = len(self.electronic_need)
        # action range for agent
        self.real_action_range = [60,80,100]
        self.possible_agents = ["player_" + str(r) for r in range(7)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        #print(self.agent_name_mapping)
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Box(low=-1,high=1,shape=(2,)) for agent in self.possible_agents}
        self.observation_spaces = {agent: Box(low= -1,high = 1,shape=(5,)) for agent in self.possible_agents}

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        # if len(self.agents) == 2:
        #     string = ("Current state: Agent1: {} , Agent2: {}".format(MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]))
        # else:
        #     string = "Game over"
        # print(string)
        pass

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.action_state = {agent: None for agent in self.agents}
        self.allocation_state={agent: None for agent in self.agents[:3]}
        self.sale_price_state= None
        self.observations = {agent: None for agent in self.agents}

        for i in self.agents:
            # print(self.agent_name_mapping[i])
            # self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            self.observations[i] = [self.range_norm(0, 200, 0),
                                    self.range_norm(0, 200, self.electronic_need[0]),
                                    self.range_norm(0, 200, 0),
                                    self.range_norm(0, 1500, 0),
                                    self.range_norm(0, 1500, 0)]
        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        if self.agent_selection == 'player_0':
            self.action_state[self.agent_selection] = np.array([action[0]*self.real_action_range[0]/2 + self.real_action_range[0]/2,
                                                action[1] * 250 + 1250 ])
        elif self.agent_selection == 'player_1':
            self.action_state[self.agent_selection] = np.array(
                [action[0] * self.real_action_range[1] / 2 + self.real_action_range[1] / 2,
                 action[1] * 250 + 1250])
        elif self.agent_selection == 'player_2':
            self.action_state[self.agent_selection] = np.array(
                [action[0] * self.real_action_range[2] / 2 + self.real_action_range[2] / 2,
                 action[1] * 250 + 1250])
        else:
            if action[0] == -1:
                self.action_state[self.agent_selection] = np.array([0, 0])
            else:
                self.action_state[self.agent_selection] = np.array([1, 1250 + action[1] * 250])


        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            # decide the sale price and allocation for every agent.
            self.allocation_price_cal()
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[self.agents[0]], self.rewards[self.agents[1]], self.rewards[self.agents[2]] = \
                 [self.allocation_state['player_0']*self.sale_price_state, self.allocation_state['player_1']*self.sale_price_state, self.allocation_state['player_2']*self.sale_price_state]

            winner = 0

            if self.action_state['player_3'][0] == 1 and self.action_state['player_4'][0] == 1:
                if self.action_state['player_3'][1] > self.action_state['player_4'][1]:
                    winner = 3
                else:
                    winner = 4
            elif self.action_state['player_3'][0] == 0 and self.action_state['player_4'][0] == 1:

                winner = 4

            elif self.action_state['player_3'][0] == 1 and self.action_state['player_4'][0] == 0:

                winner = 3

            self.rewards[self.agents[3]] = 0 if winner != 3 else \
                (self.electronic_need[self.num_moves] - 15) * self.sale_price_state - self.action_state['player_3'][1]

            self.rewards[self.agents[4]] = 0 if winner != 4 else \
                (self.electronic_need[self.num_moves] - 15) * self.sale_price_state - self.action_state['player_4'][1]

            winner = 0

            if self.action_state['player_5'][0] == 1 and self.action_state['player_6'][0] == 1:
                if self.action_state['player_5'][1] > self.action_state['player_6'][1]:
                    winner = 5
                else:
                    winner = 6
            elif self.action_state['player_5'][0] == 0 and self.action_state['player_6'][0] == 1:

                winner = 6

            elif self.action_state['player_5'][0] == 1 and self.action_state['player_6'][0] == 0:

                winner = 5

            self.rewards[self.agents[5]] = 0 if winner != 5 else \
                (self.electronic_need[self.num_moves] - 15) * self.sale_price_state * np.random.random_sample(1)[0]- self.action_state['player_5'][1]
            self.rewards[self.agents[6]] = 0 if winner != 6 else \
                (self.electronic_need[self.num_moves] - 15) * self.sale_price_state * np.random.random_sample(1)[0]- self.action_state['player_6'][1]

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= (self.max_step-1) for agent in self.agents}

            # observe the current state
            for i in self.agents:

                #print(self.agent_name_mapping[i])
                #self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
                # current state consist of last electronic need, now electronic need
                # last allocation, last price given from agent, last price given from administrator.
                self.observations[i] = [self.range_norm(0, 200, self.electronic_need[self.num_moves-1]),
                self.range_norm(0, 200, self.electronic_need[self.num_moves]),
                self.range_norm(0, 200, self.allocation_state[i]) if self.agent_name_mapping[i] < 3 else 0,
                self.range_norm(0, 1500, self.action_state[i][1]),
                self.range_norm(0, 1500, self.sale_price_state)]
                #print(self.allocation_state)
                #print(self.electronic_need[self.num_moves-1])
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            #self.state[self.agents[1 - self.agent_name_mapping[agent]]] = 0
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def allocation_price_cal(self):
        """
        decide the sale price and allocation for every agent.
        :return:
        """
        c = [self.action_state['player_0'][1], self.action_state['player_1'][1],
             self.action_state['player_2'][1]]
        #print(c)
        A = np.array([1, 1, 1]).reshape(1,3)
        b = np.array([self.electronic_need[self.num_moves]]).reshape(1,1)

        x0_bounds = [0, self.real_action_range[0]]
        x1_bounds = [0, self.real_action_range[1]]
        x2_bounds = [0, self.real_action_range[2]]

        res = linprog(c, A_eq=A, b_eq=b, bounds=[x0_bounds, x1_bounds,x2_bounds])

        # self.allocation_state[0],self.allocation_state[1],self.allocation_state[2] = \
        #     [res.x[0], res.x[1], res.x[2]]
        for agent in self.agents[:3]:
            self.allocation_state[agent] = res.x[self.agent_name_mapping[agent]]
        self.sale_price_state = max(res.x)
        return
    def range_norm(self,bottom_bound, up_bound,value):
        normed_value = (value - (bottom_bound + up_bound) / 2) / ((up_bound - bottom_bound) / 2)
        return normed_value

if __name__ == '__main__':
    env = wrapped_env()
    #api_test(env, num_cycles=10, verbose_progress=False)
    random_demo(env, render=True, episodes=1)
    # env.reset()
    # for agent in env.agent_iter():
    #     observation, reward, done, info = env.last()
    #     action = policy(observation, agent)
    #     env.step(action)