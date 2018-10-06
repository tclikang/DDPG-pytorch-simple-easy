# coding=utf-8
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class Memory(object):
    def __init__(self, capacity = 10000):
        self.capacity = capacity
        self.memory = []
        self.pos = 0  # current position that can be receive a new transition

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.pos] = Transition(*args)
        self.pos = (self.pos+1) % self.capacity

    def sample(self, batch_size):
        sample_num = min(batch_size, len(self))  # if have not enough batch_size in the memory
        return random.sample(self.memory, sample_num)

    def __len__(self):
        return len(self.memory)












