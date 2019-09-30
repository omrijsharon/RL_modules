# RL_modules
Reinforcement Learning modules for pytorch.

Action:
Choosing a *discrete* action in RL requiers many steps:
1. Getting the output of the Policy network.
2. Softmaxing it.
3. Sampling from the softmax Policy distribution.
4. Saving log_pi [log(sampled action probability)].
*One might even what to:
5. Get the chosen action in a one hot representation.
6. minimize/maximize the entropy of the Policy distribution.

*How to use?*

action = Action(PolicyNet(state))

That's it!


where PolicyNet is the policy network, state is the input for the policy network and
action is a kind of tensor with many properties:
- action.probs -> tensor of action probabilities.
- action.entropy -> tensor of the policy distribution entropy.
- action.prob -> sampled probability.
- action.log_pi -> log(sampled robability).
- action.one_hot -> one hot representation of the action.

