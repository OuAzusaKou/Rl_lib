import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_update(target, source, polyak):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))


def ddpg_step(policy_net, policy_net_target, value_net, value_net_target, optimizer_policy, optimizer_value,
              states, actions, rewards, next_states, masks, gamma, polyak):
    masks = masks.unsqueeze(-1)
    rewards = rewards.unsqueeze(-1)
    """update critic"""

    values = value_net(states, actions)

    with torch.no_grad():
        target_next_values = value_net_target(next_states, policy_net_target(next_states))
        target_values = rewards + gamma * masks * target_next_values
    value_loss = nn.MSELoss()(values, target_values)

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update actor"""

    policy_loss = - value_net(states, policy_net(states)).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """soft update target nets"""
    soft_update(policy_net_target, policy_net, polyak)
    soft_update(value_net_target, value_net, polyak)
    return value_loss, policy_loss


def critic_updator_ddpg(agent, state, action, reward, next_state, done_value, gamma, ):
    value = agent.functor_dict['critic'](state, action)
    # print(value.mean())
    with torch.no_grad():
        target_next_value = agent.functor_dict['critic_target'](next_state, agent.functor_dict['actor_target'](next_state))
        #print('reward', reward)
        #print(done_value)
        target_value = reward + gamma * (done_value) * target_next_value

    # print('done', done_value)
    # print('reward',reward.mean())
    # print('target_value', target_value[0])
    # print('value', value[0])
    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agent.optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss


def actor_updator_ddpg(agent, state, action, reward, next_state, gamma):
    for p in agent.functor_dict['critic'].parameters():
        p.requires_grad = False

    policy_loss = - agent.functor_dict['critic'](state, agent.functor_dict['actor'](state)).mean()
    # print(policy_loss)
    agent.optimizer_dict['actor'].zero_grad()
    policy_loss.backward()
    agent.optimizer_dict['actor'].step()

    for p in agent.functor_dict['critic'].parameters():
        p.requires_grad = True
    return policy_loss


def discriminator_updator(agent, state, action, label):
    '''
    judge data whether from actor or demonstrator.
    loss = CrossEntropy
    :param agent: agent object
    :param state:
    :param action:
    :param label:
    :return:
    '''
    x = state
    # print('x',x[0])
    # print('a',action[0])
    # print('l',label[0])
    for p in agent.discriminator.parameters():
        p.requires_grad = True
    z = agent.discriminator(x, action)
    #print(z.shape)

    y = torch.log(z)
    #print(y)

    loss = nn.NLLLoss()(y, label.reshape(-1).long())
    #print(loss)
    #print(agent.optimizer_discriminator)
    agent.optimizer_discriminator.zero_grad()
    loss.backward()
    # for name, parms in agent.discriminator.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:',torch.norm(parms.grad))
    agent.optimizer_discriminator.step()

    return

def actor_updator_gail(agent,state):
    '''
    update actor, the Value (Critic) is discriminator.
    :param agent:
    :param state:
    :return:
    '''
    for p in agent.discriminator.parameters():
        p.requires_grad = False

    for p in agent.actor.parameters():
        p.requires_grad = True

    policy_loss = - agent.discriminator(state, agent.actor(state))[:, 1].mean()

    #print(policy_loss)
    #print(agent.discriminator)
    agent.optimizer_actor.zero_grad()
    policy_loss.backward()

    agent.optimizer_actor.step()

    return policy_loss

def critic_updator_dqn(agent, state, action, reward, next_state, done_value, gamma, ):
    #print(action.shape)
    value = torch.gather(agent.functor_dict['critic'](state), dim=1, index=action.long())
    #print(value.shape)
    with torch.no_grad():
        next_action = torch.argmax(agent.functor_dict['critic'](next_state), dim=1).reshape((-1, 1))

        target_next_value = torch.gather(agent.functor_dict['critic'](next_state), dim=1, index=next_action.long())
        # print('reward', reward.shape)
        target_value = reward + gamma * (done_value) * target_next_value
    # print('done', done_value)
    # print('reward',reward.mean())
    # print('target_value', target_value.mean(),)
    # print('value', value.mean())
    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_dict['critic'].zero_grad()
    value_loss.backward()
    agent.optimizer_dict['critic'].step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return value_loss


def critic_updator_sac(agent, obs, action, reward, next_obs, not_done, gamma):
    with torch.no_grad():
        #print('nt', next_obs)
        _, policy_action, log_pi, _ = agent.functor_dict['actor'](next_obs)
        #print('no', next_obs)
        target_Q1, target_Q2 = agent.functor_dict['critic_target'](next_obs, policy_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - agent.functor_dict['log_alpha'].exp().detach() * log_pi
        target_Q = reward + (not_done * gamma * target_V)

    # get current Q estimates
    current_Q1, current_Q2 = agent.functor_dict['critic'](
        obs, action)
    critic_loss = F.mse_loss(current_Q1,
                             target_Q) + F.mse_loss(current_Q2, target_Q)



    # Optimize the critic
    agent.optimizer_dict['critic'].zero_grad()
    critic_loss.backward()
    agent.optimizer_dict['critic'].step()

    # for name, parms in agent.functor_dict['critic'].named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)


def actor_and_alpha_updator_sac(agent, obs, target_entropy):
    # detach encoder, so we don't update it with the actor loss
    _, pi, log_pi, log_std = agent.functor_dict['actor'](obs)
    actor_Q1, actor_Q2 = agent.functor_dict['critic'](obs, pi)

    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (agent.functor_dict['log_alpha'].exp().detach() * log_pi - actor_Q).mean()

    entropy = 0.5 * log_std.shape[1] * \
        (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)


    # optimize the actor
    agent.optimizer_dict['actor'].zero_grad()
    actor_loss.backward()
    agent.optimizer_dict['actor'].step()


    agent.optimizer_dict['log_alpha'].zero_grad()
    alpha_loss = (agent.functor_dict['log_alpha'].exp() *
                  (-log_pi - target_entropy).detach()).mean()

    alpha_loss.backward()
    agent.optimizer_dict['log_alpha'].step()
