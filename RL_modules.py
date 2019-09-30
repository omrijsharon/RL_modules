import numpy as np
import copy
import torch
from torch.distributions import Categorical


def np2torch(x):
    return torch.from_numpy(x).float()


def idx2one_hot(idx, length):
    x = torch.zeros(length)
    x[idx] = 1
    return x


class Action:
    def __init__(self, p, manual_sample_idx=-1):
        self.p = Categorical(logits=p)
        self.probs = self.p.probs.view(-1)
        self.entropy = self.p.entropy().view(-1)
        self.action_dict = {'probs': self.probs, 'entropy': self.entropy}
        if manual_sample_idx == -1:
            # sample:
            self.idx = self.p.sample()
        elif 0 <= manual_sample_idx < len(self.probs):
            self.idx = manual_sample_idx
        else:
            raise Exception("Error in Action: manual sample index mush be between 0 to" + str(len(self.probs)) + " not including the last.")
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

    def push(self, action):
        for key, value in action.action_dict.items():
            self.action_dict[key] = torch.cat((self.action_dict[key], value.view(1,-1)), dim=0)
        self.probs = self.action_dict['probs']
        self.entropy = self.action_dict['entropy']
        self.idx = self.action_dict['index']
        self.log_pi = self.action_dict['log_prob']
        self.prob = self.action_dict['prob']
        self.one_hot = self.action_dict['one_hot']

    def push_list(self, actions_list):
        for action in actions_list:
            self.push(action)


class RewardMemory:
    def __init__(self, running_step=50):
        self.history = np.array([])
        self.mean = np.array([])
        self.std = np.array([])
        self.running_step = running_step

    def reset(self):
        self.__init__()

    def push(self, reward):
        self.history = np.append(self.history, reward)
        self.mean = np.append(self.mean, self.history[-self.running_step:].mean())
        self.std = np.append(self.std, self.history[-self.running_step:].std())

    def push_list(self, reward_list):
        for reward in reward_list:
            self.push(reward)

    def calc_discount_rewards(self, reward_episode, gamma=0.99, normalize=True):
        rewards = copy.deepcopy(reward_episode)
        for i in range(1, len(rewards)):
            rewards[-i - 1] += gamma * rewards[-i]
        if normalize is True:
            rewards -= rewards.mean()
            rewards /= (rewards.std() + np.finfo(np.float32).eps)
        return rewards
