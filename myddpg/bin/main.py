# coding=utf-8
'''
Plesase read our code refer to Algorithm1 of DDPG(CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING)
'''
from utils.inithyperpara import init_hyper_para
import utils.network as net
import torch
import copy
from utils.memory import Memory
from utils.randomprocess import OUNoise
import gym
import torch.optim as opt
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import gc
import os
from graphviz import Digraph
import numpy as np
import utils.network
import torch.onnx
from utils.visulize import make_dot
from torch.autograd import Variable

# because the game will never be 'done', so we set each episode have 200 steps
# to make our model learn faster.
MAX_STEP = 200

# set run model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def update_critic_net(state, next_state, reward, action,
                      target_actor_net, target_critic_net,
                      critic_net,optimizer_critic,args):
    #  mu'(s_{i+1}). see Algorithm1 in DDPG
    next_action = target_actor_net(next_state).detach()  # shape:[128,1]
    # Q'(s_{i+1},mu'(s_{i+1}))  see Algorithm1 in DDPG
    next_state_reward = target_critic_net(next_state, next_action).detach()  # shape:[128,1]
    # y_i, see Algorithm1 in DDPG
    y = (reward + args.sigma * next_state_reward).detach()
    # state should be detach, otherwise will throw an runtime error
    # RuntimeError: Trying to backward through the graph a second time,
    # but the buffers have already been freed. Specify retain_graph=True
    # when calling backward the first time. 
    # To understand this error, see: https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
    q = critic_net(state.detach(), action.detach())
    # the loss function. see L in Algorithm1 of DDPG
    loss = F.smooth_l1_loss(q, y)
    # you can set a breakpoint below and use make_dot(loss).view() to see the net structure.
    print(['critic loss', loss])
    optimizer_critic.zero_grad()
    loss.backward()
    # loss.backward()
    optimizer_critic.step()


def update_actor_net(state, actor_net, critic_net, optimizer_actor):
    # mu(s_i)
    new_action = actor_net(state.detach())
    # Q(s_i, mu(s_i))
    new_reward = critic_net(state.detach(), new_action)
    # using the Q value as the loss of actor net.
    # attention: there is a -1 to make it as a loss which should be reduce every iteration
    loss = -1 * new_reward.mean()  # 注意前面有个负号
    print(['actor loss', loss])
    optimizer_actor.zero_grad()
    loss.backward()
    # loss.backward()
    optimizer_actor.step()


# main 函数，程序入口
def main():
    # initialize the game
    env = gym.make('Pendulum-v0').unwrapped
    print env.observation_space
    print env.observation_space.high
    print env.observation_space.low
    print env.action_space
    # import hyper parameters
    args = init_hyper_para()
    # random initialize critic network
    state_dim = env.reset().shape[0]
    action_dim = env.action_space.shape[0]
    #  if we have the saved model, load it
    if os.path.exists('/home/likang/PycharmProjects/myddpg/bin/Models/critic.ckpt'):
        critic_net = torch.load('/home/likang/PycharmProjects/myddpg/bin/Models/critic.ckpt')
    else:  # initialize the model
        critic_net = net.CriticNetwork(state_dim=state_dim, action_dim=action_dim).to(device)  # need to init paras according to the gym game
    # random initialize actor network(also called policy network)
    if os.path.exists('/home/likang/PycharmProjects/myddpg/bin/Models/actor.ckpt'):
        actor_net = torch.load('/home/likang/PycharmProjects/myddpg/bin/Models/actor.ckpt')
    else:
        actor_net = net.ActorNetwork(state_dim=state_dim, action_dim=action_dim).to(device)
    # initialize
    optimizer_critic = opt.Adam(critic_net.parameters(), lr=0.001)
    optimizer_actor = opt.Adam(actor_net.parameters(), lr=0.001)
    # initialize target critic network which is the same of critic network
    target_critic_net = copy.deepcopy(critic_net)
    # initialize target actor network which is the same of actor network
    target_actor_net = copy.deepcopy(actor_net)
    # init the memory buffer
    memory = Memory(args.capacity)
    # initialize a random process N for action exploration
    ounoise = OUNoise(env.action_space.shape[0])  # init random process
    # enter circle of training process
    for ep in range(args.num_ep):
        print(["ep: ", ep])
        # reset random process
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                          ep) / args.exploration_end + args.final_noise_scale
        ounoise.reset()
        # initialize a state s1
        state = env.reset()  # 这里把state初始化成二维的tensor
        state = torch.tensor([state], dtype=torch.float32).to(device)
        for t in range(MAX_STEP):
            print(['time step: ', t])
            # select a action according to actor network(also called policy network)
            action = actor_net.select_action(state, ounoise)
            # execute the action and get a new state s_i+i
            # get a reward from the environment
            next_state, reward, done, _ = env.step([action.item()])
            # store the transition {s_i, a_i, r_i, s_i+1} into memory
            next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
            reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
            memory.push(state, action, reward, next_state)
            state = next_state
            # print([state, action, reward, next_state])
            del action, reward, next_state
            # get a batch_size transitions.
            # (s_i, a_i, r_i, s_{i+1}) in Algorithm1 of DDPG
            transitions = memory.sample(args.batch_size)
            s1 = torch.cat([tran.state for tran in transitions])
            s2 = torch.cat([tran.next_state for tran in transitions])
            r1 = torch.cat([tran.reward for tran in transitions])
            a1 = torch.cat([tran.action for tran in transitions])
            update_critic_net(s1, s2, r1, a1,
                              target_actor_net, target_critic_net,
                              critic_net, optimizer_critic,args)
            # update actor policy network
            update_actor_net(s1, actor_net, critic_net, optimizer_actor)
            # update target critic network
            # theta^{Q'}, see algorithm1 of DDPG
            for target_param, source_param in zip(target_critic_net.parameters(), critic_net.parameters()):
                target_param.data.copy_(args.tau*source_param + (1-args.tau) * target_param)
            # update target actor network
            # theta^{mu'}, see algorithm1 of DDPG
            for target_param, source_param in zip(target_actor_net.parameters(), actor_net.parameters()):
                target_param.data.copy_(args.tau*source_param + (1-args.tau) * target_param)
            # show image
            plt.imshow(env.render('rgb_array'))
            time.sleep(0.001)
            # finish
            if done:
                break
            del transitions
        gc.collect()

        if ep % 10 == 0:  # save model
            torch.save(critic_net, './Models/' + 'critic.ckpt')
            torch.save(actor_net, './Models/' + 'actor.ckpt')


if __name__ == '__main__':
    main()