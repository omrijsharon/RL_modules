# RL_modules
Reinforcement Learning modules for pytorch.

**Requirements:**
PyTorch 1.1

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
- An Action object is getting the linear output of the Policy network by:
```
import RL_modules as rl


action = rl.Action(PolicyNet(state))
```
and then automatically does steps 2-6 which are contained in the Action object.


### How to use it in a gym environment?
```
import RL_modules as rl


#get the linear output of the Policy network
action = rl.Action(PolicyNet(state))

#give the environment a sampled action
next_state, reward, done, info = env.step(action())
```
*That's it!*

where PolicyNet is the policy network, state is the input for the policy network and
action is an object containing useful information about the action:
- action.probs -> tensor of action probabilities.
- action.entropy -> policy distribution entropy.
- **action()** = aciton.idx -> sampled action index.
- action.prob -> sampled probability.
- action.log_pi -> log(sampled probability).
- action.one_hot -> one hot representation of the action.

One can append action in a list or push it into ActionMemory class for later training:

## ActionMemory module

You can push an action into ActionMemory as a list of actions or as a single action not in a list.

Examples for implementation:

1st example - push a list of actions:
```
import RL_modules as rl

# begining of the code:
actionMemory = rl.ActionMemory()
action_list = []

# middle of the code:
action = rl.Action(PolicyNet(state))
action_list.append(action)

# end of the code:
actionMemory.push(action_list)
```

2nd example - push an action:
```
import RL_modules as rl

#begining of the code:
actionMemory = rl.ActionMemory()

#middle of the code:
action = rl.Action(PolicyNet(state))
actionMemory.push(action)
```

