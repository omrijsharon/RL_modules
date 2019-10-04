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
    idx_range = torch.arange(idx.view(-1).size(0))
    x = torch.zeros(len(idx_range), length)
    x[idx_range, idx] = 1
    return x


def normalize(x):
    x -= x.mean()
    x /= (x.std() + np.finfo(np.float32).eps)
    return x


def is_grad_fn(x, grad_fn='Softmax'):
    if hasattr(x, 'grad_fn'):
        result = grad_fn.lower() in (str(x.grad_fn)).lower()
    else:
        result = False
    return result


def is_distribution(x):
    if hasattr(x, 'sum'):
        result = torch.prod((x.sum(-1) - 1).abs() < 1e-7).item() is 1
    else:
        result = False
    return result


class PGloss(nn.Module):
    def __init__(self):
        super(PGloss, self).__init__()

    def forward(self, log_pi, discounted_rewards):
        return (-log_pi*discounted_rewards).sum()


class Action:
    def __init__(self, pi):
        if len(pi) > 0 and pi.__class__.__name__ is "Tensor":
            softmax_check = is_grad_fn(pi, grad_fn='Softmax')
            if pi.dim() == 1:
                pi = pi.unsqueeze(0)
            if softmax_check:
                p = Categorical(probs=pi)
            else:
                if is_distribution(pi):
                    p = Categorical(probs=pi)
                else:
                    p = Categorical(logits=pi)
            self.probs = p.probs.view(pi.size(0), -1)
            self.logits = p.logits.view(pi.size(0), -1)
            self.entropy = p.entropy().view(pi.size(0), -1)
            self.idx = p.sample().view(-1)
            self.log_prob = p.log_prob(self.idx).view(-1)
            self.prob = p.probs[torch.arange(len(self.idx)), self.idx].view(-1)
            self.one_hot = idx2one_hot(self.idx, pi.size(-1))
        else:
            self.probs = torch.tensor([])
            self.logits = torch.tensor([])
            self.entropy = torch.tensor([])
            self.idx = torch.LongTensor([])
            self.log_prob = torch.tensor([])
            self.prob = torch.tensor([])
            self.one_hot = torch.tensor([])

        b = dir(self).index('entropy')
        n = b + 7
        self.keys = dir(self)[b:n]

    def __add__(self, other):
        if other.__class__.__name__ is "Action":
            if other.size(1) == self.size(1) or other.size(1) == 0 or self.size(1) == 0:
                self.push_mem(other)
            else:
                raise RuntimeError("invalid argument: Sizes of Actions must match except in dimension 0. Got " + str(other.size(1)) + " and " + str(self.size(1)) + " in dimension 1.")
        else:
            if other.__class__.__name__ is "Tensor":
                raise TypeError("You tried to combine Action with Tensor. Convert the Tensor to Action by: Action(tensor) before adding it to Action.")
            else:
                raise TypeError("Action can only be combined with another Action. You tried to combine Action with " + other.__class__.__name__ + ".")
        return self

    def push_mem(self, action):
        for key in self.keys:
            self.__setattr__(key, torch.cat((self.__dict__[key], action.__dict__[key]), dim=0))

    def push(self, action):
        self.__add__(action)

    def append(self, action):
        self.__add__(action)

    def size(self, i=None):
        if self.probs.size()[0] > 0:
            s = tuple([j for j in self.probs.size()])
            if i is not None:
                s = s[i]
        else:
            s = (0, 0)
            if i is not None:
                s = s[i]
        return s

    def __getitem__(self, action_idx):
        action = Action([])
        if "int" in action_idx.__class__.__name__:
            for key in self.keys:
                action.__setattr__(key, torch.unsqueeze(self.__getattribute__(key)[action_idx],0))
        else:
            for key in self.keys:
                action.__setattr__(key, self.__getattribute__(key)[action_idx])
        return action


    def __call__(self, i=None):
        if i is None:
            result = self.idx[-1].item()
        else:
            result = self.idx[i].item()
        return result

    def __repr__(self):
        return 'Action' + str(self.idx.view(-1))[6:] + ' Size' + str(self.size())

    def __len__(self):
        return len(self.idx)


x = Action([])
print(x)
x += Action(torch.randn((20, 5), requires_grad=True))
print(x)


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
