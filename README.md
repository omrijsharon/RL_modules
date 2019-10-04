# RL_modules
Reinforcement Learning modules for pytorch.

#### Table of content
- [x] Policy Gradient Loss
- [x] Action
- [x] StateMemory
- [x] RewardMemory
- [x] RewardHistory
- [ ] MemoryManager


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
    while env.done is False:
        actions += rl.Action(PolicyNet(state))
        next_state, reward, done, info = env.step(action())
        rewards = np.append(rewards, reward)
    discount_r = calc_discount_rewards(rewards, gamma=0.99, norm=True)    
    loss = loss_func(actions.log_prob, discount_r) - beta*actions.log_prob.sum()
    PolicyNet.optimizer.zero_grad()
    loss.backward()
    PolicyNet.optimizer.step()

```
