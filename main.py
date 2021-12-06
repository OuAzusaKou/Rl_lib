
from absl import flags
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions
import sys
from pysc2.env import sc2_env
from pysc2.env import environment
from gym import spaces
import gym
import numpy as np
# This is a sample Python script.
FLAGS = flags.FLAGS
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_MOVE_SCREEN = actions.FUNCTIONS.Move_Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_SMART_MINIMAP = actions.FUNCTIONS.Smart_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]

N_DISCRETE_ACTIONS = 512
N_CHANNELS = 2
HEIGHT = 16
WIDTH = 16
step_mul = 16
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Box(low=-0.01, high=0.01, shape=(2, ),
                                                                         dtype=np.float32)))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        #print(action)
        #order = action // 256
        #print('realaction', action)
        order = action[0]
        #coord = [action % 256 // 16, action % 256 % 16]
        coord = [action[1]*800+8, action[2]*800+8]
        # print(coord)
        if _MOVE_SCREEN not in self.obs[0].observation["available_actions"]:
            self.obs = sc2env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
        # print(_MOVE_SCREEN)
        if order == 1:
            # print('action_z',action_z)
            new_action = [sc2_actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, coord])]
        elif order == 0:
            # print('action_z',action_z)
            new_action = [sc2_actions.FunctionCall(_MOVE_CAMERA, [coord])]
        else:
            raise ('action error')

        self.obs = sc2env.step(actions=new_action)
        # buf_action = [
        #  sc2_actions.FunctionCall(_MOVE_CAMERA, [np.random.randint(low=0, high=64, size=(2,)).tolist()])]
        # buf_obs = env.step(actions=buf_action)
        player_relative = self.obs[0].observation["feature_screen"][_PLAYER_RELATIVE]

        feature_map = (player_relative == _PLAYER_NEUTRAL).astype(int)

        mini_map = self.obs[0].observation["feature_minimap"][_PLAYER_RELATIVE].astype(int)

        mini_map = (mini_map == _PLAYER_NEUTRAL).astype(int)

        new_screen = np.stack((feature_map, mini_map), axis=0)

        observation = new_screen

        reward = float(self.obs[0].reward)
        # print(reward)
        done = self.obs[0].step_type == environment.StepType.LAST

        info = {}
        return observation, reward, done, info

    def reset(self):
        obs = sc2env.reset()
        # Select all marines first
        obs = sc2env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

        self.obs = obs

        player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]

        feature_map = (player_relative == _PLAYER_NEUTRAL).astype(int)

        mini_map = obs[0].observation["feature_minimap"][_PLAYER_RELATIVE].astype(int)

        mini_map = (mini_map == _PLAYER_NEUTRAL).astype(int)

        screen = np.stack((feature_map, mini_map), axis=0)

        observation = screen

        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    FLAGS(sys.argv)
    AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(screen=16, minimap=16))

    sc2env = sc2_env.SC2Env(
        map_name="FindAndCollect",
        players=[sc2_env.Agent(sc2_env.Race.protoss)],
        step_mul=step_mul,
        visualize=False,
        agent_interface_format=AGENT_INTERFACE_FORMAT
    )
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
