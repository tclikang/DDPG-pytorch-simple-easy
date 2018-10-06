# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from randomprocess import OUNoise
from graphviz import Digraph
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # set gpu model

class CriticNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, h1_dim=256, h2_dim=128, h3_dim = 128):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc1_action = nn.Linear(action_dim,h2_dim)
        self.fc3 = nn.Linear(h2_dim+h2_dim, h3_dim)  # output only one Q-value
        self.fc4 = nn.Linear(h3_dim,action_dim)


    def forward(self, state, action):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))  # 128
        action_out = F.relu(self.fc1_action(action))
        out = torch.cat([out, action_out], 1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class ActorNetwork(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, h1_dim=256, h2_dim=128,h3_dim=64):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, h3_dim)
        self.fc4 = nn.Linear(h3_dim, action_dim)


    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        # action space is [-2 +2], so we should *2.0
        out = F.tanh(self.fc4(out)) * 2.0
        return out

    def select_action(self, state, noise, train=True):
        if train is True:
            # all the action value are normalized into [-1, +1]
            action = self.forward(state) + 2.0*torch.tensor(noise.noise(), device=device, dtype=torch.float32)
        else:  # 不是训练，直接选择动作
            action = self.forward(state)
        # the action space of this game [-2 +2], so we should limit the action.
        return action.clamp(-2.0, 2.0)










































