import numpy as np
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt


def np2torch(x):
    return torch.from_numpy(x).float()


def idx2one_hot(idx, length):
    x = torch.zeros(length)
    x[idx] = 1
    return x

def normalize(x):
    x -= x.mean()
    x /= (x.std() + np.finfo(np.float32).eps)
    return x


class PGloss(nn.Module):
    def __init__(self):
        super(PGloss, self).__init__()

    def forward(self, log_pi, discounted_rewards):
        return (-log_pi*discounted_rewards).sum()


class Action:
    def __init__(self, p, sample_size=(1,)):
        self.p = Categorical(logits=p)
        self.probs = self.p.probs.view(-1)
        self.entropy = self.p.entropy().view(-1)
        self.action_dict = {'probs': self.probs, 'entropy': self.entropy}
        self.idx = self.p.sample(sample_size)
        self.log_pi = self.p.log_prob(self.idx)
        self.prob = self.probs[self.idx]
        self.one_hot = idx2one_hot(self.idx, len(self.probs))
        self.sample_dict = {'index': self.idx, 'prob': self.prob, 'log_prob': self.log_pi, 'one_hot': self.one_hot}
        self.action_dict.update(self.sample_dict)

    def __repr__(self):
        return 'Action ' + str(self.action_dict)

    def __call__(self, *args, **kwargs):
        return self.idx


class ActionMemory:
    def __init__(self):
        self.action_dict = {'probs': torch.tensor([]), 'entropy': torch.tensor([])}
        self.sample_dict = {'index': torch.LongTensor([]), 'prob': torch.tensor([]), 'log_prob': torch.tensor([]), 'one_hot': torch.tensor([])}
        self.action_dict.update(self.sample_dict)

    def reset(self):
        self.__init__()

    def push_mem(self, action):
        for key, value in action.action_dict.items():
            self.action_dict[key] = torch.cat((self.action_dict[key], value.view(1,-1)), dim=0)
        self.probs = self.action_dict['probs']
        self.entropy = self.action_dict['entropy']
        self.idx = self.action_dict['index']
        self.log_pi = self.action_dict['log_prob']
        self.prob = self.action_dict['prob']
        self.one_hot = self.action_dict['one_hot']

    def push(self, action):
        if 'Action' in str(type(action)):
            self.push_mem(action)
        else:
            for a in action:
                self.push_mem(a)

    def __len__(self):
        return len(self.idx)


class RewardMemory:
    def __init__(self):
        self.reward_episode = np.array([])

    def reset(self):
        self.__init__()

    def push(self, reward):
        self.reward_episode = torch.cat((self.reward_episode, reward.view(1,-1)), dim=0)

    def __call__(self, *args, **kwargs):
        self.discount_rewards = self.calc_discount_rewards(*args, **kwargs)
        return np2torch(self.discount_rewards)

    def calc_discount_rewards(self, gamma=0.99, norm=True):
        rewards = copy.deepcopy(self.reward_episode)
        for i in range(1, len(rewards)):
            rewards[-i - 1] += gamma * rewards[-i]
        if norm is True:
            rewards = normalize(rewards)
        return rewards

    def plot(self, color='blue', fontsize=18):
        steps_axis = np.arange(len(self.reward_episode))
        plt.plot(steps_axis, self.reward_episode, color=color)
        plt.fill_between(steps_axis, np.zeros(len(steps_axis)), self.discount_rewards, color=color, linewidth=0.0, alpha=0.5)
        plt.xlabel('Step number', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)


class StateMemory:
    def __init__(self):
        self.state = torch.tensor([])
        self.state_ = torch.tensor([])# next_state
        self.done = torch.BoolTensor([])

    def reset(self):
        self.__init__()

    def push(self, state=None, state_=None, done=None):
        if state is not None:
            self.state = torch.cat((self.state, np2torch(state).view(1, -1)), dim=0)

        if state_ is not None:
            self.state_ = torch.cat((self.state_, np2torch(state_).view(1, -1)), dim=0)

        if done is not None:
            self.done = torch.cat((self.done, torch.BoolTensor([done]).view(1, -1)), dim=0)


class RewardHistory:
    '''
    * Mostly for plotting.
    '''
    def __init__(self, running_step=50):
        self.history = np.array([])
        self.mean = np.array([])
        self.std = np.array([])
        self.running_step = running_step
        self.episode_axis = np.array([])

    def reset(self):
        self.__init__()

    def push_mem(self, reward):
        self.history = np.append(self.history, reward)
        self.mean = np.append(self.mean, self.history[-self.running_step:].mean())
        self.std = np.append(self.std, self.history[-self.running_step:].std())
        self.episode_axis = np.append(self.episode_axis, len(self.history))

    def push(self, reward):
        if 'list' not in str(type(reward)):
            self.push_mem(reward)
        else:
            for r in reward:
                self.push_mem(r)

    def plot(self, color='orange', fontsize=18):
        plt.plot(self.episode_axis, self.history, '.', color=color, alpha=0.5, label='Reward')
        plt.plot(self.episode_axis, self.mean, color=color, label='Average')
        plt.fill_between(self.episode_axis, self.mean + self.std, self.mean - self.std, color=color, linewidth=0.0, alpha=0.3)
        plt.xlabel('Episode number', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)
        plt.legend()

    def __len__(self):
        return len(self.history)

class MemoryManager:
    def __init__(self):
        self.memory = {}
        self.memory['State'] = StateMemory()
        self.memory['Action'] = ActionMemory()
        self.memory['Reward'] = RewardMemory()
        # self.memory['State'] = torch.tensor([])
        # self.memory['Action'] = torch.tensor([])
        # self.memory['State_'] = torch.tensor([])
        # self.memory['Reward'] = torch.tensor([])
        # self.memory['Done'] = torch.tensor([])

    def reset(self):
        self.__init__()

    def push(self, state=None, action=None, state_=None, reward=None, done=None):
        if state is not None:
            self.memory['State'].push(state=state)

        if action is not None:
            self.memory['Action'].push(action=action)

        if state_ is not None:
            self.memory['State'].push(state_=state_)

        if reward is not None:
            self.memory['Reward'].push(reward=reward)

        if done is not None:
            self.memory['State'].push(done=done)
