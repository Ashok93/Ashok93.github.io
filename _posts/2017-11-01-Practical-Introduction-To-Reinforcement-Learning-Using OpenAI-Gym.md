---
layout: post
title: Practical Introduction to Reinforcement Learning Using OpenAI `gym`
---

Machine Learning enthusiasts might already have heard or known about Reinforcement Learning(RL). For those who don't, here is a brief introduction. Reinforcement Learning is the problem of getting an agent to act in the world so as to maximize its rewards. For example, consider a dog, we cannot directly tell the dog what it has to do but we can train the dog by giving it rewards(like food) if it does what is expected in the world and punish(do not give food) for not accomplishing the required task. Slowly the dog will figure out what it did to get the reward. Similarly, we can train a computer to do several tasks in this fashion, and it has been found that this technique can solve many problems which are difficult to solve in the traditional way very well. Few examples where RL is currently being applied - Playing Games like Chess, Go, video games, controlling robot manipulators, etc...

This post will be a gentle practical introduction to RL. We are going to get our hands dirty by trying out RL in OpenAI's `gym` environment. This post is intented to bring an intution about how RL works and have the environment set for further experimentation. We are going to take a simple example from `gym` environment `CartPole-v0`.  Here is the wiki about the cart-pole <https://github.com/openai/gym/wiki/CartPole-v0>. Please have the link handy in another tab.

As we can see that the `gym` environment has observation and action. `Observation` is the observation made once an action is taken on the environment. They depict the current state of the enviroment after some action is taken. Imagine it to be the sensor information of the robot after some action is taken by the robot. `Action` is the action that the agent can perform on the environment. Imagine it to be moving the robot arm using actuators. `reward` is the reward that is given for every action that is made. Generally, the reward function is defined by the user. I hope this gives an abstract idea of what are the various terminologies that we are going to use and their significance.


### Installing gym

```python
#if you have pip, use
pip install gym #for python 2.7
pip3 install gym #for python 3
```

To check if you have it installed,

```python
#in the python try
import gym
env = gym.make('CartPole-v0')
```

if you donot find any errors:
 	congrats! 
else:
	Please refer <https://gym.openai.com/docs/> for installation if you have any issues.

