# RL_modules
Reinforcement Learning modules for pytorch.

#### Table of content
- [x] Policy Gradient Loss
- [ ] Entropy Loss
- [x] Action
- [ ] Reward
- [ ] RewardHistory
- [ ] MemoryManager


**Requirements:**
- PyTorch 1.1
- numpy 1.16

## What is it good for?
### Solving openAI gym CartPole in less than 30 code lines with RL_modules
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
Updating us every 10 episodes with the cumulative reward of the last episode:
```
if episode%10 == 0:
    print("Episode #", episode, " score: ", rewards.sum().item())
```
Plotting the raw cumulative reward, its running mean and its running standard deviation:
```
reward_history.plot()
plt.show()
```

## Policy Gradient loss function
The gradient of the loss function is defined by:
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla_{\theta}J=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log(\pi_{\theta})Q^{\pi_{\theta}}(s,a)]" title="\Large \nabla_{\theta}J=\mathbb{E}_{\pi_{\theta}}[\nabla_{\theta}log(\pi_{\theta})Q^{\pi_{\theta}}(s,a)]" />

### Using PGloss

Similar to using any pytorch loss function, we declare the loss function in the begining and use it later. i.e.:
```
import RL_modules as rl

#beginning of the code
loss_func = rl.PGloss()

#backprop:
loss = loss_func(log_pi, Q)
```

-_IMPORTANT_: This function causes the gradients to **accent** (as they should) when using any optimizer for gradient descent. So use the function 'as is'.


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
- action[-5:] -> a new action object with the last 5 actions only.
- action[b:n] -> a new action object with actions b to n.
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

### Coding example:
```
import numpy as np
import RL_modules as rl
import gym


env = gym('')
PolicyNet = Network()
loss_func = rl.PGloss()
beta = 5e-2 #entropy loss coefficient

for episode in range(n_episodes)
    actions = rl.Action([])
    rewards = np.array([])
    state = env.reset()
    while env.done is False:
        actions += rl.Action(PolicyNet(state))
        next_state, reward, done, info = env.step(action())
        rewards = np.append(rewards, reward)
    discount_r = calc_discount_rewards(rewards, gamma=0.99, norm=True)    
    loss = loss_func(actions.log_prob, discount_r) - beta*actions.entropy.sum()
    PolicyNet.optimizer.zero_grad()
    loss.backward()
    PolicyNet.optimizer.step()

```
