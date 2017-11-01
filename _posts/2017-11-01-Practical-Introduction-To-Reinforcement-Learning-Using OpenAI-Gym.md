---
layout: post
title: Practical Introduction to Reinforcement Learning Using OpenAI `gym`
---

![an image alt text]({{ site.baseurl }}/images/cartpole.gif "Cart Pole")

Machine Learning enthusiasts might already have heard or known about Reinforcement Learning(RL). For those who don't, here is a brief introduction. Reinforcement Learning is the problem of getting an agent to act in the world so as to maximize its rewards. For example, consider a dog, we cannot directly tell the dog what it has to do but we can train the dog by giving it rewards(like food) if it does what is expected in the world and punish(do not give food) for not accomplishing the required task. Slowly the dog will figure out what it did to get the reward. Similarly, we can train a computer to do several tasks in this fashion, and it has been found that this technique can solve many problems which are difficult to solve in the traditional way very well. Few examples where RL is currently being applied - Playing Games like Chess, Go, video games, controlling robot manipulators, etc...

This post will be a gentle practical introduction to RL. We are going to get our hands dirty by trying out RL in OpenAI's **`gym`** environment. This post is intented to bring an intution about how RL works and have the environment set for further experimentation. We are going to take a simple example from **`gym`** environment **`CartPole-v0`**.  Here is the wiki about the cart-pole <https://github.com/openai/gym/wiki/CartPole-v0>. Please have the link handy in another tab.

As we can see that the **`gym`** environment has observation and action. **`Observation`** is the observation made once an action is taken on the environment. They depict the current state of the enviroment after some action is taken. Imagine it to be the sensor information of the robot after some action is taken by the robot. **`Action`** is the action that the agent can perform on the environment. Imagine it to be moving the robot arm using actuators. **`reward`** is the reward that is given for every action that is made. Generally, the reward function is defined by the user. I hope this gives an abstract idea of what are the various terminologies that we are going to use and their significance.


## INSTALLING GYM

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

## APPROACH

The carpole challenge is to prevent the pendulum from falling over. It is a very simple challenge in openAI environment.

First we will start with a random approach, where we try to solve the cartpole problem with random set of parameters. Then we will try using a hill climb approach to figure out the optimum parameter. Finally we can make use of DQN(Deep Q Network) to solve the same problem.


## CODE

### Random Approach

We first will try using a random approach to try to solve the problem. In cart pole example, there are four observables/state values. We first randomly initialize four parameters, multiply with the observables/state and then decide based on the result to take one of two discrete actions(push left or right). We do this for a number of episodes, say 1000, and then see the reward that we get for the action that we make. In cart pole, the problem is considered solved if the reward is >= 200. Thus we can use this a termination point to break the loop to find the optimum set of parameters.

The code for random approach is given below.

```python
import numpy as np
import gym

def run_episode(env, params):
	observation = env.reset() #this has the env measurables such as angle, pos of cart etc. 4 values/params
	total_reward = 0
	
	for _ in range(200):
		env.render()
		action = 0 if np.matmul(params, observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		total_reward += reward

		if done:
			break

	return total_reward

def run_successful_episode(env, params):
	observation = env.reset()

	while True:
		env.render()
		action = 0 if np.matmul(params, observation) < 0 else 1
		observation, reward, done, info = env.step(action)

		if done:
			print("Balanced the pole successfully")
			break


def train_model():
	env = gym.make('CartPole-v0')
	best_reward = 0
	best_param = None
	counter = 0

	for _ in range(10000):
		counter += 1
		parameters = np.random.rand(env.observation_space.shape[0])
		reward = run_episode(env, parameters) #compute reward for that particular set of random params

		if reward > best_reward:
			best_reward = reward
			best_param = parameters

		print("best reward at ", counter, "iteration is", best_reward)		

		if reward == 200:
			break

	return counter, best_param

if __name__ == '__main__':
	no_of_tries, params = train_model()

	print('Total tries made by the program to find optimum parameter for balancing is ', no_of_tries)
	print('The parameters for successful control are ', params)
	print('Running successful solution....')

	env = gym.make('CartPole-v0')
	run_successful_episode(env, params)	

```

If you run the code, we can see that the random approach does quite well for this problem almost solving it easily finding optimum parameters in less than 20 episodes.

