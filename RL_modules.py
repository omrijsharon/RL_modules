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


def calc_discount_rewards(reward_episode, gamma=0.99, norm=True):
    rewards = copy.deepcopy(reward_episode)
    for i in range(1, len(rewards)):
        rewards[-i - 1] += gamma * rewards[-i]
    if norm is True:
        rewards = normalize(rewards)
    return rewards


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
        if "Action" in log_pi.__class__.__name__:
            log_pi = log_pi.log_prob
        if "Reward" in discounted_rewards.__class__.__name__:
            discounted_rewards = discounted_rewards()
        return (-log_pi*discounted_rewards).sum()


class CLIPloss(nn.Module):
    '''PPO loss function'''
    def __init__(self, epsilon=0.2):
        super(CLIPloss, self).__init__()
        self.epsilon = epsilon

    def forward(self, action_old, action, advantage, epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon
        r = action.prob/action_old.prob
        return (torch.min(r*advantage, torch.clamp(r, 1-self.epsilon, 1+self.epsilon))*advantage).sum()


class Entropyloss(nn.Module):
    def __init__(self):
        super(Entropyloss, self).__init__()

    def forward(self, action):
        return -action.entropy.sum()


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
            self.entropy = p.entropy().view(-1)
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


class Reward:
    def __init__(self, reward=None):
        if reward is not None:
            if 'int' in reward.__class__.__name__ or 'float' in reward.__class__.__name__:
                self.reward = torch.tensor([1.*reward])
            elif len(reward)==0:
                self.reward = torch.tensor([])
            else:
                self.reward = np2torch(reward)
        else:
            self.reward = torch.tensor([])

    def __add__(self, other):
        if 'Reward' in other.__class__.__name__:
            self.reward = torch.cat((self.reward, other.reward.view(-1)), dim=0)
        else:
            if 'Tensor' in other.__class__.__name__:
                raise TypeError("You tried to combine Reward with Tensor. Convert the Tensor to Reward by: Reward(tensor) before adding it to Reward.")
            else:
                raise TypeError("Reward can only be combined with another Reward. You tried to combine Reward with " + other.__class__.__name__ + ".")
        return self

    def push(self, reward):
        self.__add__(reward)

    def append(self, reward):
        self.__add__(reward)

    def __getitem__(self, action_idx):
        reward = Reward([])
        reward.reward = self.reward[action_idx]
        return reward

    def __call__(self, gamma=0.99, norm=True):
        self.discount_rewards = calc_discount_rewards(self.reward, gamma=gamma, norm=norm)
        return self.discount_rewards

    def sum(self):
        return self.reward.sum()

    def __repr__(self):
        return 'Reward' + str(self.reward)[6:] + ' Size' + str(self.size())

    def size(self):
        if self.reward.size()[0] > 0:
            s = tuple([j for j in self.reward.size()])
        else:
            s = (0, 0)
        return s

    def __len__(self):
        return len(self.reward)

    def plot(self, color='blue', fontsize=12):
        steps_axis = np.arange(len(self.reward))
        plt.plot(steps_axis, self.reward.numpy(), color=color)
        plt.fill_between(steps_axis, np.zeros(len(steps_axis)), self.discount_rewards.numpy(), color=color, linewidth=0.0, alpha=0.5)
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
    def __init__(self, running_steps=50):
        self.history = np.array([])
        self.mean = np.array([])
        self.std = np.array([])
        self.running_step = running_steps
        self.episode_axis = np.array([])

    def reset(self):
        self.__init__()

    def __add__(self, other):
        if 'Reward' in other.__class__.__name__:
            self.history = np.append(self.history, other.sum().numpy())
            self.mean = np.append(self.mean, self.history[-self.running_step:].mean())
            self.std = np.append(self.std, self.history[-self.running_step:].std())
            self.episode_axis = np.append(self.episode_axis, len(self.history))
        else:
            if 'Tensor' in other.__class__.__name__:
                raise TypeError("RewardHistory object cannot contain a Tensor. Convert the Tensor to Reward by: Reward(tensor) before containing it in a RewardHistory object.")
            else:
                raise TypeError("RewardHistory can contain only Reward objects, but you tried to contain " + other.__class__.__name__ + ".")
        return self

    def append(self, reward):
        self.__add__(reward)

    def push(self, reward):
        if 'list' not in str(type(reward)):
            self.__add__(reward)
        else:
            for r in reward:
                self.__add__(r)

    def plot(self, color='orange', fontsize=12):
        plt.plot(self.episode_axis, self.history, '.', markeredgewidth=0, color=color, alpha=0.4, label='Reward')
        plt.plot(self.episode_axis, self.mean, color=color, label='Average')
        plt.fill_between(self.episode_axis, self.mean + self.std, self.mean - self.std, color=color, linewidth=0.0, alpha=0.3)
        plt.xlabel('Episode number', fontsize=fontsize)
        plt.ylabel('Reward', fontsize=fontsize)
        plt.legend()

    def __len__(self):
        return len(self.history)

    def __repr__(self):
        return 'RewardHistory' + str(self.episode_axis) + ' Size' + str(self.__len__())

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


class RND:
    def __init__(self, RNDnet, PRDnet, memory_capacity=2000):
        '''
        :param RNDnet: Random Network which will remain frozen.
        :param PRDnet: Predictor Network that will learn the RNDnet output.
        :param memory_capacity: number of samples/state that will be stored and learned from.
        parameters for optimizer should be called:
            RND.PRDnet.parameters()
        '''
        self.RNDnet = RNDnet
        self.PRDnet = PRDnet
        self.learn_on_spot = hasattr(self.PRDnet, 'optimizer')
        if self.learn_on_spot is False:
            self.optimizer_setup()
        self.memory_capacity = memory_capacity
        self.mem = torch.tensor([])

    def __call__(self, input):
        '''
        :param input:
        https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/
        Random Network Distillation (RND) *recommended:
            The authors recommend the input to be obs(t+1) to overcome the 3 prediction error factors.
        Next-State Prediction (NSP) *less-recommended:
            Using input as cat(obs(t), action) makes it less resilient to the noisy-TV-Problem.
        :return:
            curiosity reward for each step
        '''
        if len(input.size()) < 2:
            input = input.view(1, -1)
        self.mem = torch.cat((self.mem, input), dim=0)
        with torch.no_grad():
            self.RNDnet.eval()
            self.PRDnet.eval()
            self.RND_reward = torch.sum((self.RNDnet(input) - self.PRDnet(input)) ** 2, dim=1).detach()
        return self.RND_reward

    def backward(self, chunk_size=100):
        self.RNDnet.eval()
        self.PRDnet.train()
        chunks = [self.mem[i:i + chunk_size, :] for i in range(0, len(self.mem), chunk_size)]
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                retain_graph = True
            else:  # last chunk
                retain_graph = False
            with torch.no_grad():
                RND_out = self.PRDnet(self.mem)
            PRD_out = self.PRDnet(self.mem)
            self.RND_loss = ((RND_out - PRD_out) ** 2).sum()
            self.RND_loss.backward(retain_graph=retain_graph)
        mem_diff = len(self.mem) - self.memory_capacity
        if mem_diff > 0:
            idx = torch.randperm(len(self.mem))[mem_diff:]
            self.mem = self.mem[idx]

    def learn(self, n_epochs=1, chunk_size=100):
        if self.learn_on_spot:
            for epoch in range(n_epochs):
                self.PRDnet.optimizer.zero_grad()
                self.backward(chunk_size=chunk_size)
                self.PRDnet.optimizer.step()
        else:
            raise Exception(
                'Error in RND->learn: PRDnet has no optimizer within it. Add an optimizer attribute with an optimizer to PRDnet to use this function.')

    def optimizer_setup(self, optimizer_name='Adam', lr=1e-3, weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay
        if 'Adam' in optimizer_name:
            self.PRDnet.optimizer = optim.Adam(self.PRDnet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif 'RMSprop' in optimizer_name:
            self.PRDnet.optimizer = optim.RMSprop(self.PRDnet.parameters(), lr=self.lr, weight_decay=self.weight_decay, alpha=0.99, eps=1e-8, momentum=0, centered=False)
