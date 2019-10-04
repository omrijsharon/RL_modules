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
```
- **action()** -> last sampled action index.
- action.probs -> tensor of action probabilities.
- action.entropy -> policy distribution entropy.
- action.idx -> sampled action indices.
- action.prob -> sampled probability.
- action.log_pi -> log(sampled probability).
- action.one_hot -> one hot representation of the sampled actions.

- action(0) -> sampled index of the 1st action.
- action(-1) -> sampled index of the last action (equivalent to action()).
- action(n) -> sampled index of the n-th action.
#### Similar to numpy.array and torch.tensor:
- action([]) -> an empty action.
- action[-5:] -> a new action object with the last 5 actions only.
- action[b:n] -> a new action object with actions b to n.
#### Adding/appending/pushing new actions to an existing Action:
As long as new_action is an Action object, you combine between actions in the following ways (all methods are equivalent):
- action = action + new_action
- action += new_action
- action.append(new_action)
- action.push(new_action)

