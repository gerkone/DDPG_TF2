# DDPG_TF2
It was hard to find a simple and tidy DDPG implementation in TF2, so I made one. 

## DDPG
DDPG is an model-free, off-policy algorithm that learns a Q-function and a policy in a **continuous** action space. It is inspired by Deep Q Learning, and can be seen as DQN on a continuous acion space.
It employs the use of off-policy data and the Bellman equation to learn the Q function which is in turn used to derive and learn the policy.

The DDPG algorith was originaly described [this paper](https://arxiv.org/pdf/1509.02971.pdf).

#### Main features:
- Stochastic (deep) model estimation allows for continuous (infinite) action spaces.
- Use of a **noise process** (for example the _Ornstein–Uhlenbeck_ process) for action space exploration.
- Use of **experience replay** for a stable learning on previous experiences.
- Actor and critic structure
- Use of target models for both actor and critic networks (weight transfer with Polyak averaging).
- Use of the Bellman equation to describe the optimal q-value function for each pair <state, action>.

## Performance on OpenAI gym environments
### Pendulum-v0
The model usually needs about 70-80 iterations to reach a decent performance. 
This number may be decreased by further hyperparameter tuning.

Performance after 70 iterations:

![Pendulum-v0](https://media2.giphy.com/media/731NWtJGS7onYIAqgN/giphy.gif)
