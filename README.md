# RL_modules
Reinforcement Learning modules for pytorch.

## Table of content
### Losses:
- [x] Policy Gradient Loss
- [x] CLIP loss (PPO)
- [x] Entropy Loss

### Objects:
- [x] Action - discrete (Categorical/Multinomial)
- [ ] Action - continuous (Normal/OUNoise (https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py))
- [x] Reward - no documentation!!!
- [x] RewardHistory - no documentation!!!
- [ ] MemoryManager
- [x] RND (Random Network Distillation) - adding curiosity to your agent
- [ ] GAE (general advantages estimator)


### What is it good for? with a code example:
- [x] Solving openAI gym CartPole in less than 30 lines of code using RL_modules
- [x] Breaking down the code



**Requirements:**
- PyTorch 1.1
- numpy 1.16


## Policy Gradient loss function
The gradient of the loss function is defined by:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla_{\theta}J=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log(\pi_{\theta})Q^{\pi_{\theta}}(s,a)]" title="\Large \nabla_{\theta}J=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log(\pi_{\theta})Q^{\pi_{\theta}}(s,a)]" />

### Using PGloss
Similar to using any pytorch loss function, we declare the loss function in the begining and use it later. i.e.:
```
import RL_modules as rl

#beginning of the code
PGloss_func = rl.PGloss()

#backprop:
loss = PGloss_func(log_pi, Q)
#or
loss = PGloss_func(actions.probs, Q)
#or
loss = PGloss_func(actions, Q)
#or
loss = PGloss_func(actions, rewards)
```
where log_pi and actions.probs are the log of the sampled actions probability, actions is an Action object and rewards is a Reward object.


-_IMPORTANT_: This function causes the gradients to **accent** (as they should) when using any optimizer for gradient descent. So use the function 'as is'.

## CLIP loss function (PPO)
<img src="https://latex.codecogs.com/svg.latex?\Large&space;J^\text{CLIP}(\theta)=\mathbb{E}[\min(r(\theta)\hat{A}_{\theta_\text{old}}(s,a),\text{clip}(r(\theta),1-\epsilon,1+\epsilon)\hat{A}_{\theta_\text{old}}(s,a))]" title="\Large J^\text{CLIP} (\theta) = \mathbb{E} [ \min( r(\theta) \hat{A}_{\theta_\text{old}}(s, a), \text{clip}(r(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{\theta_\text{old}}(s, a))]" />

A summarizes of the PPO algorithm:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#ppo

### Using CLIPloss
Example:
```
import RL_modules as rl

CLIPloss_func = rl.CLIPloss() #option A
#or
CLIPloss_func = rl.CLIPloss(epsilon=0.1) #option B

#backprop:
loss = CLIPloss_func(action_old, action, advantage, epsilon=0.3) #goes well with option A
#or
loss = CLIPloss_func(action_old, action, advantage) #goes well with option B
#or
loss = CLIPloss_func(action_old, action, advantage, epsilon=0.3) #with option B: epsilon=0.1 from initialization will be ignored
#or
loss = CLIPloss_func(actions, rewards) #with option A: default epsilon=0.2 will be used.
```
where
- action_old is the output Action object  of <img src="https://latex.codecogs.com/svg.latex?\Large&space;\pi_{\theta_\text{old}}(a\vert{s})" title="\Large \pi_{\theta_\text{old}}(a\vert{s})" />
- action is the output Action object  of <img src="https://latex.codecogs.com/svg.latex?\Large&space;\pi_{\theta}(a\vert{s})" title="\Large \pi_{\theta}(a\vert{s})" />
- advantage usually is the output of the old critic model.

## Entropy loss function
The entropy loss tries to **maximize** the entropy of the policy distribution to increase exploration.

### Using Entropyloss
Combining PGloss and Entropy loss. example:
```
import RL_modules as rl

#beginning of the code
PGloss_func = rl.PGloss()
Entropyloss_func = rl.Entropyloss()

#backprop:
loss = PGloss_func(actions, Q) + beta * Entropyloss_func(actions)
```

- beta is a regularization factor for the Entropy loss.

## Action module

Choosing a *discrete* action in RL requiers many steps:
1. Getting  the linear output of the Policy network.
2. Softmaxing it.
3. Sampling from the softmax Policy distribution.
4. Saving log_pi [log(sampled action probability)].
5. Get the entropy of the Policy distribution for later minimization/maximization.
6. Get the chosen action in a one hot representation.

#### To make our lives easier, I coded the Action module:
An Action object is like a numpy.array or a torch.tensor specially tailored for reinforcement learning. 

Just convert the output of the Policy network to an Action by:
```
import RL_modules as rl

y = PolicyNet(state)
action = rl.Action(y) #converting PolicyNet output to an Action
```
and Action will execute and save the results of steps 2-6.

Action automatically checks if the output of the Policy network is **linear or a distribution** and acts accordingly.


### How to use it in a gym environment?
```
import RL_modules as rl

#begining of the code: initializing an empty Action object. 
action = Action([])


#Convert the Policy network's output to an Action, and add the new Action to the cumulative Action.
action += rl.Action(PolicyNet(state))

#outputs the last sampled action to the environment.
next_state, reward, done, info = env.step(action())
```
*That's it!*

where PolicyNet is the policy network, state is the policy network's input and
action is an object containing useful information for training.
### Syntax
- **action()** -> last sampled action index.
- action.probs -> a tensor of action probabilities.
- action.entropy -> a tensor of policy distribution entropy.
- action.idx -> a tensor of sampled action indices.
- action.prob -> a tensor of sampled probability.
- action.one_hot -> a tensor of sampled actions' one hot representation.
- action.log_prob -> a tensor of log(sampled probability). This is the famous  <img src="https://latex.codecogs.com/svg.latex?\Large&space;log(\pi_{\theta})" title="\Large log(\pi_{\theta})" />  from the Policy Gradient Loss which is very useful when training an agent with PG method.

#### Getting sampled actions:
- action(0) -> sampled index of the 1st action.
- action(-1) -> sampled index of the last action (equivalent to action()).
- action(n) -> sampled index of the n-th action.
#### Indexing (similar to numpy.array and torch.tensor):
- action([]) -> an empty action.
- action[-5:] -> a new Action object with the last 5 actions only.
- action[b:n] -> a new Action object with actions b to n.
#### Size and length:
- action.size() -> a tuple with the number of sampled actions in index 0 and the number of possible actions in index 1.
- action.size(n) -> n-th index of action.size().
- len(action) -> the same as action.size(0).
#### Adding/appending/pushing new actions to an existing Action:
To combine between action and new_action, both must be Action objects (all combination methods are equivalent):
- action = action + new_action
- action += new_action
- action.append(new_action)
- action.push(new_action)

i.e.:
```
import RL_modules as rl


x = rl.Action(torch.randn(5))
print(x.size())
---> (1, 5)
x += rl.Action(torch.randn((20, 5)))
print(x.size())
---> (21, 5)
```


## RND (Random Network Distillation)
The goal of RND module is to add curiosity to your agent.
It can also be used to find anomalies in data, recognize a familiar pathes and much more.
A good explanation about the algorithm can be found here:
https://towardsdatascience.com/reinforcement-learning-with-exploration-by-random-network-distillation-a3e412004402

### Using RND
Before calling RND module, you must initialize 2 networks (architecture issues are discussed in the next part):
1. RNDnet - a network which will remain frozen and will only be used for calculations. No gradients will pass in this network.
2. PRDnet - a predictor network which will try to guess the output of the RND network and learn how to be more like RNDnet.
- _IMPORTANT_: It is recomended to add your own optimizer to PRDnet in advance. i.e.:
```
PRDnet.optimizer = optim.Adam(PRDnet.parameters(), lr=1e-3)
```
Initializing (after RNDnet and PRDnet exist):
```
import RL_modules as rl


rnd = RND(RNDnet, PRDnet, memory_capacity = 5000)
```
Then use one of the two:
1. Getting the intrinsic RND curiosity reward (immuned to the noisy-TV problem):
```
next_state = rl.np2torch(next_state)
intrinsic_reward = rnd(next_state)
```
_OR_

2. Getting Next-State Prediction (NSP) reward (NOT immuned to the noisy-TV problem):
```
state = rl.np2torch(state)
intrinsic_reward = rnd(torch.cat((state, action.one_hot), dim=-1))
```
### Architecture
The RNDnet and PRDnet can have the same architecture (it is recommended) but they must be initialized with different weights.
#### What should be the dimensions of the input/output of the networks?
The networks must have the same input and output dimensions. The number of dimensions may differ only in the hidden layers.
##### Input dimensions
Notice from the last part that:
1. The input to the RND module is a state. If you use RND curiosity reward, the networks' input dimension should be the same as the state dimension.
2. The input to the RND module is a state + action. If you use NSP reward, the networks' input dimension should be the sum of the state and action dimensions.
##### Output dimensions
You can choose the output dimensions of the RNDnet and PRDnet with no constraints. Yet, you must remember:
1. The length of a diagonal of a 1x1 square is <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sqrt{2}" title="\Large \sqrt{2}" />. The length of a diagonal of a 1x1x1 cube is <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sqrt{3}" title="\Large \sqrt{3}" />. The higher the dimention, the more volume the shape has and the farther away random points within the shape are.
2. Higher dimentsion in the output layer might slow down the calculations and the learning process.

A recommended number of dimentions for the output is 2-4.



## What is it good for?
### Solving openAI gym CartPole in less than 30 lines of code using RL_modules
Using REINFORCE algorithm + plotting the results. Example:
```
import torch
import gym
from NetworkModule import Network
import RL_modules as rl
import matplotlib.pyplot as plt


env = gym.make('CartPole-v1') #create the environment
#Parameters and Hyperparameters
n_episodes = 200
lr = 1e-2 #learning_rate
beta = 1e-6 #entropy loss coefficient
gamma = 0.99 #discount factor

torch.manual_seed(41)
PolicyNet = Network(L=[4,*1*[8],2], lr=lr, optimizer='RMSprop', dropout=0)
PolicyNet.PGloss = rl.PGloss()
PolicyNet.Entropyloss = rl.Entropyloss()
reward_history = rl.RewardHistory(running_steps=10)
for episode in range(n_episodes):
    state = env.reset()
    actions = rl.Action([])
    rewards = rl.Reward([])
    done = False
    while done is False:
        actions += rl.Action(PolicyNet(rl.np2torch(state)))
        next_state, reward, done, info = env.step(actions())
        rewards.append(rl.Reward(reward))
        state = next_state
        env.render()
    loss = PolicyNet.PGloss(actions, rewards(gamma, norm=True)) + beta*PolicyNet.Entropyloss(actions)
    PolicyNet.optimizer.zero_grad()
    loss.backward()
    PolicyNet.optimizer.step()
    reward_history += rewards
    if episode%10 == 0:
        print("Episode #", episode, " score: ", rewards.sum().item())
reward_history.plot()
plt.show()
```
![alt text](https://github.com/omrijsharon/RL_modules/blob/master/Figure_1_cartpole.png)

### Let's break down the code and understand every part of it:

Creating  the CartPole environment from gym package:
```
env = gym.make('CartPole-v1') #create the environment
```
Defining parameters and hyperparametes:
```
lr = 1e-2 #learning_rate
beta = 1e-6 #entropy loss coefficient
gamma = 0.99 #discount factor
n_episodes = 200 #number of episodes
```
Initializing a policy network using NetworkModule(https://github.com/omrijsharon/NetworkModule).
The Policy network suppose to get a state as an input and outputs a number for each action (that becomes actions distribution later on).
The network architecture:
- an input layer with 4 nodes - because the state shape is 4.
- 1 hidden layer with 8 nodes - these numbers are hyperparameters.
- an output layer with 2 nodes: - because the action space has 2 discrete actions.
```
torch.manual_seed(41)
PolicyNet = Network(L=[4,*1*[8],2], lr=lr, optimizer='RMSprop', dropout=0)
```
Setting the loss functions:
```
PolicyNet.PGloss = rl.PGloss()
PolicyNet.Entropyloss = rl.Entropyloss()
```
Initializing a RewardHistory module. This module will save the cumulative reward of each episode. It will calculate the mean and the standard deviation of last 10 cumulative rewards (running mean and running std).
```
reward_history = rl.RewardHistory(running_steps=10)
```
Looping over the environment for n_episodes and initializing the environment:
```
for episode in range(n_episodes):
    state = env.reset()
```
Initializing empty Action and Reward modules:
```
    actions = rl.Action([])
    rewards = rl.Reward([])
```
Setting done to False and starting to interact with the environment. The environment sets done to be True in the end of an episode.
```
    done = False
        while done is False:
```
The following bulleted commands are all contained in this line:
```
        actions += rl.Action(PolicyNet(rl.np2torch(state)))
```
- np2torch converts a numpy array to a torch tensor: 
```
state_tensor = rl.np2torch(state_array)
```
- Running tensored state in the Policy network and outputing a tensor with size 2.
```
output_tensor = PolicyNet(state_tensor)
```
- Converting the Policy network output to an Action object and adding/appending it to the last actions
```
actions += rl.Action(output_tensor)
```

actions() returns an integer which represents the last sampled action.
env.step(actions()) gives the envirenment an action and gets its response as a state and reward.
if done is True, the episode will end.
state = next_state updates the state for the next loop.
```
        next_state, reward, done, info = env.step(actions())
        rewards.append(rl.Reward(reward))
        state = next_state
```
Rendering the environment. For faster preformance, delete this line or make it a comment with #.
```
env.render()
```
rewards(gamma, norm=True) returns normalized dicounted reward.

PGloss gets 2 arguments.
1. log_pi tensor or actions.log_prob or just an Action object.
PGloss knows to handle with an Action object and get its log_prob attribute automatically.
2. dicounted_rewards tensor or rewards(gamma, norm=True) or rewards.
PGloss knows to handle with a Reward object. Using only rewards (without the brackets) will result using default arguments.

Entropyloss gets an Action object and returns -action.entropy.sum(). Minimization of this loss results in more exploration and less certainty in the actions the agents is choosing. This effect will increase with beta.
```
loss = PolicyNet.PGloss(actions, rewards(gamma, norm=True)) + beta*PolicyNet.Entropyloss(actions)
```
zeroing grads, backpropagating the loss, walking one step in the gradient direction:
```
    PolicyNet.optimizer.zero_grad()
    loss.backward()
    PolicyNet.optimizer.step()
```
Adding/appending the cumulative reward from the last episode to a RewardHistory object so we can plot it and see the learning progress of the agent:
```
reward_history += rewards
```
Updating every 10 episodes with the cumulative reward of the last episode:
```
if episode%10 == 0:
    print("Episode #", episode, " score: ", rewards.sum().item())
```
Plotting the raw cumulative reward, its running mean and its running standard deviation:
```
reward_history.plot()
plt.show()
```
