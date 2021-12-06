import torch
import torch.nn as nn


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
    value = agent.critic(state, action)
    # print(value.mean())
    with torch.no_grad():
        target_next_value = agent.critic_target(next_state, agent.actor_target(next_state))
        # print('reward', reward.shape)
        target_value = reward + gamma * (done_value) * target_next_value
    # print('done', done_value)
    # print('reward',reward.mean())
    # print('target_value', target_value.mean(),)
    # print('value', value.mean())
    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_critic.zero_grad()
    value_loss.backward()
    agent.optimizer_critic.step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return


def actor_updator_ddpg(agent, state, action, reward, next_state, gamma):
    for p in agent.critic.parameters():
        p.requires_grad = False

    policy_loss = - agent.critic(state, agent.actor(state)).mean()
    # print(policy_loss)
    agent.optimizer_actor.zero_grad()
    policy_loss.backward()
    agent.optimizer_actor.step()

    for p in agent.critic.parameters():
        p.requires_grad = True
    return


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
    value = torch.gather(agent.critic(state), dim=1, index=action.long())
    #print(value.shape)
    with torch.no_grad():
        next_action = torch.argmax(agent.critic(next_state), dim=1).reshape((-1, 1))

        target_next_value = torch.gather(agent.critic(next_state), dim=1, index=next_action.long())
        # print('reward', reward.shape)
        target_value = reward + gamma * (done_value) * target_next_value
    # print('done', done_value)
    # print('reward',reward.mean())
    # print('target_value', target_value.mean(),)
    # print('value', value.mean())
    value_loss = nn.MSELoss()(value, target_value)
    # print('loss',value_loss)
    agent.optimizer_critic.zero_grad()
    value_loss.backward()
    agent.optimizer_critic.step()
    # for name, parms in agent.critic.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)

    return