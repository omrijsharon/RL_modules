# RL_modules
Reinforcement Learning modules for pytorch.

Action:
Choosing a *discrete* action in RL requiers many steps:
1. Getting  the linear output of the Policy network.
2. Softmaxing it.
3. Sampling from the softmax Policy distribution.
4. Saving log_pi [log(sampled action probability)].

*One might even what to:*
5. Get the chosen action in a one hot representation.
6. minimize/maximize the entropy of the Policy distribution.


*How to use it?*

action = Action(PolicyNet(state))
state, reward, done, info = env.step(action())

That's it!

where PolicyNet is the policy network, state is the input for the policy network and
action is now an object with many properties:
- action.probs -> tensor of action probabilities.
- action.entropy -> policy distribution entropy.
- aciton.idx -> sampled action index.
- action.prob -> sampled probability.
- action.log_pi -> log(sampled probability).
- action.one_hot -> one hot representation of the action.

One can append action in a list or push it into ActionMemory class for later training.
