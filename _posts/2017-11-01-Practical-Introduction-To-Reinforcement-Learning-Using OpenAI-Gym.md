---
layout: post
title: A Practical Introduction to Reinforcement Learning Using OpenAI `gym`
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

First we will start with a random approach, where we try to solve the cartpole problem with random set of parameters. We will then implement a DQN(Deep Q Network) for the same problem.

## CODE

### Random Approach

We first will try using a random approach to try to solve the problem. In cart pole example, there are four observables/state values. We first randomly initialize four parameters, multiply with the observables/state and then decide based on the result to take one of two discrete actions(push left or right). We do this for a number of episodes, say 1000, and then see the reward that we get for the action that we make. In cart pole, the problem is considered solved if the reward is >= 200. Thus we can use this a termination point to break the loop to find the optimum set of parameters.

The code for random approach is given below.

First we define a `train_model` function, where we create a gym environment and run loop for some episodes(say 10000). For every episode, we randomly pick 4 parameters(There are 4 observables in our problem) and we run the episode using `run_episode` function to calculate the reward. If we can get the reward of 200 then we can consider the problem solved. The code snippet is provided below.

```python
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
```


The `run_episode` function takes the random parameters as input and iterates (200) times and takes action on the environment based on the parameters and observation(`np.matmul(params, observation)`). The function returns the total reward. 

```python
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

```

The entire code for this can be found in my github link: <https://github.com/Ashok93/OpenAI-Cartpole/blob/master/cartpole-random.py>

If you run the code, we can see that the random approach does quite well for this problem almost solving it easily finding optimum parameters in less than 20 episodes.


### Deep Q Network

The random approach was pretty simple and effective in solving this particualar problem. But if the complexity of the problem increases, also if the problem tends to get more non linear, then you will see that our random approach will never be able to predict the optimal parameter. In these cases, we switch to more sophosticated algorithms. One such algorithm in reinforcement learning is the Deep Q Network.

Deep Q Learning is basically Q Learning algorithm applied to the deep learning. I dont want to get into details of deep learning as this is out of the scope of the article. I am planning to write an article on the neural networks and deep learning soon. 

Q-learning is a model free reinforcement learning technique. The main idea of Q learning is to find a optimal action-selection policy for a given process(Markov Decision Process). A policy is a rule that the agent follows in selecting actions, given the state it is in. When such an action-value function is learned, the optimal policy can be constructed by simply selecting the action with the highest value in each state.

Here is the wiki link for the Q learning algo: <https://en.wikipedia.org/wiki/Q-learning>

We will use Keras as deep learning library as it is super simple to build DNN models using them. I am using tensor flow backend.

If you are not familiar with neural networks and their working, consider the neural network as a black box containing one giant optimization problem or finding some patterns between input and output. This post is only intended to apply and see RL algorithms in action. 

We will be using a NN with one input layer and three hidden layers. The image below shows only two hidden layers but we will be using three. There are two outputs(0 and 1) depicting the action to be made in the game. 

Using Keras, the model implementation is as simples as code shown below.

```python
model = Sequential()
model.add(Dense(24, input_dim=self.state_size, activation = 'relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(self.action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
```

Here we just say that our model has two hidden layer of 24 nodes and one output layer having two nodes(`self.action_size`). In order of NN to learn from the data, we use `fit` method passing it the input and output. Something like `model.fit(state, reward)`. After training the model, we can use the `predict` function, to predict the reward for the given state. From this you can see that the NN studies/interprets the pattern between the input and the output.

