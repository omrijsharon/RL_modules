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
import RL_modules as RL

#beginning of the code
loss_func = RL.PGloss()

#backprop:
loss = loss_func(log_pi, Q)
```

-_IMPORTANT_: This function causes the gradients to **accent**, as they should. So use the function as is.


## Action

Choosing a *discrete* action in RL requiers many steps:
1. Getting  the linear output of the Policy network.
2. Softmaxing it.
3. Sampling from the softmax Policy distribution.
4. Saving log_pi [log(sampled action probability)].

*One might even what to:*

5. Get the chosen action in a one hot representation.
6. minimize/maximize the entropy of the Policy distribution.


### How to use it in a gym environment?
```
action = Action(PolicyNet(state))
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

## ActionMemory

You can push an action into ActionMemory as a list of actions or as a single action not in a list.

Examples for implementation:

1st example - push a list:
```
# begining of the code:
actionMemory = ActionMemory()
action_list = []

# middle of the code:
action = Action(PolicyNet(state))
action_list.append(action)

# end of the code:
actionMemory.push(action_list)
```

2nd example - push an action:
```
#begining of the code:
actionMemory = ActionMemory()

#middle of the code:
action = Action(PolicyNet(state))
actionMemory.push(action)
```

