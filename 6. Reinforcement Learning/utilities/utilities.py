# -*- coding: utf-8 -*-


# """
# installing dependencies
# """
# !apt-get -qq -y install libnvtoolsext1 > /dev/null
# !ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.8.0 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so
# !apt-get -qq -y install xvfb freeglut3-dev ffmpeg> /dev/null
# !pip -q install gym
# !pip -q install pyglet
# !pip -q install pyopengl
# !pip -q install pyvirtualdisplay

"""
Imports
"""

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
import random
from gym import wrappers

# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1024, 768))
# display.start()
import os
# os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)

# import matplotlib.animation
import numpy as np
# from IPython.display import HTML

import torch
import torchvision

# import torch.utils.tensorboard as tb

from PIL import Image

from torch.utils import data 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import glob, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""# Homework 6 - Imitation and Reinforcement Learning

### Getting to know OpenAI Gym. (TODO)

We will be using the OpenAI Gym as our environment -- **we strongly recommend looking over the ["Getting Started" documentation](https://gym.openai.com/docs/) .**


> A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.


![](https://user-images.githubusercontent.com/8510097/31701297-3ebf291c-b384-11e7-8289-24f1d392fb48.PNG)


>The goal position is 0.5, the location of the flag on top of the hill.

>Reward: -1 for each time step, until the goal position of 0.5 is reached.

>Initialization: Random position from -0.6 to -0.4 with no velocity.

>Episode ends when you reach 0.5 position, or if 200 timesteps are reached. (So failure to reach the flag will result in a reward of -200).
"""

class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):

        from PIL import Image
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)



def dummy_policy(env, num_episodes):

  '''
  TODO: Fill in this function 

  Functionality: This should be executing a random policy sampled from the action space of the environment for num_episodes long
                  and should be returning the mean reward over those episodes and the frames of the rendering recorded on the last episode 
  
  Input: env, the MountainCar environment object 
         num_episodes, int, the total number of episodes you want to run this for 

  Returns: mean_reward, float, which is the mean_reward over num_episodes 
           frames, a list, which should contain elements of image dimensions (i.e RGB, with size that you specify), should have a length of the last episode that you record.

  '''
  total_reward = 0
  frames = []

  for i in range(num_episodes):
    # Initial environment state
    state = env.reset()

    for t in range(200):
      # Render environment for last episode
      if i == (num_episodes - 1):
        # frames.append(env.render(mode='rgb_array'))
        frames.append(ResizeObservation.render(env))
      # Determine next action
      action = env.action_space.sample()
      # Get next_state, reward, and done using env.step()
      state, reward, done, info = env.step(action)
      # Update total reward
      total_reward += reward
      # Break if done
      if done:
        # print("Episode finished after {} timesteps".format(t+1))
        break

  # Compute mean reward
  mean_reward = total_reward / (num_episodes * 200)

  return mean_reward, frames

resize_observation_shape = 100
env = gym.make('MountainCar-v0')
env = ResizeObservation(env, resize_observation_shape)

# rew, frames = dummy_policy(env, 10)

# #### Video plotting code ######################
# plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
# patch = plt.imshow(frames[0])
# plt.axis('off')
# animate = lambda i: patch.set_data(frames[i])
# ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
# HTML(ani.to_jshtml())

"""## Expert Reinforcement Learning Code - Q-Learning


You are given the code for training a traditional Q-learning based agent. Please go through this code.

### Supporting functions
"""

def discretize(state, discretization, env):

    env_minimum = env.observation_space.low
    state_adj = (state - env_minimum)*discretization
    discretized_state = np.round(state_adj, 0).astype(int)

    return discretized_state


def choose_action(epsilon, Q, state, env):
    """    
    Choose an action according to an epsilon greedy strategy.
    Args:
        epsilon (float): the probability of choosing a random action
        Q (np.array): The Q value matrix, here it is 3D for the two observation states and action states
        state (Box(2,)): the observation state, here it is [position, velocity]
        env: the RL environment 
        
    Returns:
        action (int): the chosen action
    """
    action = 0
    if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state[0], state[1]]) 
    else:
        action = np.random.randint(0, env.action_space.n)
  
    return action


def update_epsilon(epsilon, decay_rate):
    """
    Decay epsilon by the specified rate.
    
    Args:
        epsilon (float): the probability of choosing a random action
        decay_rate (float): the decay rate (between 0 and 1) to scale epsilon by
        
    Returns:
        updated epsilon
    """
  
    epsilon *= decay_rate

    return epsilon


def update_Q(Q, state_disc, next_state_disc, action, discount, learning_rate, reward, terminal):
    """
    
    Update Q values following the Q-learning update rule. 
    
    Be sure to handle the terminal state case.
    
    Args:
        Q (np.array): The Q value matrix, here it is 3D for the two observation states and action states
        state_disc (np.array): the discretized version of the current observation state [position, velocity]
        next_state_disc (np.array): the discretized version of the next observation state [position, velocity]
        action (int): the chosen action
        discount (float): the discount factor, may be referred to as gamma
        learning_rate (float): the learning rate, may be referred to as alpha
        reward (float): the current (immediate) reward
        terminal (bool): flag for whether the state is terminal
        
    Returns:
        Q, with the [state_disc[0], state_disc[1], action] entry updated.
    """    
    if terminal:        
        Q[state_disc[0], state_disc[1], action] = reward

    # Adjust Q value for current state
    else:
        delta = learning_rate*(reward + discount*np.max(Q[next_state_disc[0], next_state_disc[1]]) - Q[state_disc[0], state_disc[1],action])
        Q[state_disc[0], state_disc[1],action] += delta
  
    return Q

"""#### Wrapper for Rendering the Environment"""

class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):

        from PIL import Image
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)

"""### Main Q-learning Loop"""

def Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes=5000):
    """
    
    The main Q-learning function, utilizing the functions implemented above.
          
    """
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached 
    frames = []
  
    for i in range(max_episodes):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = env.reset() # initial environment state
        state_disc = discretize(state,discretization,env)

        while done != True:                 
            # Determine next action 
            action = choose_action(epsilon, Q, state_disc, env)                                      
            # Get next_state, reward, and done using env.step(), see http://gym.openai.com/docs/#environments for reference
            if i==1 or i==(max_episodes-1):
              frames.append(env.render())
            next_state, reward, done, _ = env.step(action) 
            # Discretize next state 
            next_state_disc = discretize(next_state,discretization,env)
            # Update terminal
            terminal = done and next_state[0]>=0.5
            # Update Q
            Q = update_Q(Q,state_disc,next_state_disc,action,discount,learning_rate, reward, terminal)  
            # Update tot_reward, state_disc, and success (if applicable)
            tot_reward += reward
            state_disc = next_state_disc

            if terminal: success +=1 
            
        epsilon = update_epsilon(epsilon, decay_rate) #Update level of epsilon using update_epsilon()

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state[0])
        success_list.append(success/(i+1))

        if (i+1) % 100 == 0:
            print('Episode: ', i+1, 'Average Reward over 100 Episodes: ',np.mean(reward_list))
            reward_list = []
                
    env.close()
    
    return Q, position_list, success_list, frames

"""### Define Params and Launch Q-learning"""

# # Initialize Mountain Car Environment
env = gym.make('MountainCar-v0')

env = ResizeObservation(env,100) #Resize observations

env.seed(42)
np.random.seed(42)
env.reset()

# Parameters    
learning_rate = 0.2 
discount = 0.9
epsilon = 0.8 
decay_rate = 0.95
max_episodes = 5000
discretization = np.array([10,100])


# #InitQ
# num_states = (env.observation_space.high - env.observation_space.low)*discretization
# #Size of discretized state space 
# num_states = np.round(num_states, 0).astype(int) + 1
# # Initialize Q table
# Q = np.random.uniform(low = -1, 
#                       high = 1, 
#                       size = (num_states[0], num_states[1], env.action_space.n))

# # Run Q Learning by calling your Qlearning() function
# Q, position, successes, frames = Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes)

# np.save('./expert_Q.npy',Q) #Save the expert

"""### Visualization

#### Plotting
"""

# import pandas as pd 

# plt.plot(successes)
# plt.xlabel('Episode')
# plt.ylabel('% of Episodes with Success')
# plt.title('% Successes')
# plt.show()
# plt.close()

# p = pd.Series(position)
# ma = p.rolling(3).mean()
# plt.plot(p, alpha=0.8)
# plt.plot(ma)
# plt.xlabel('Episode')
# plt.ylabel('Position')
# plt.title('Car Final Position')
# plt.show()

"""#### Agent's Video"""

# #### Video plotting code #####################
# deep_frames = []
# for obs in frames:
#   im = Image.fromarray(np.uint8(obs))
#   im = im.resize((600,400))
#   deep_frames.append(np.asarray(im))

# plt.figure(figsize=(deep_frames[0].shape[1] / 72.0, deep_frames[0].shape[0] / 72.0), dpi = 72)
# patch = plt.imshow(deep_frames[0])
# plt.axis('off')
# animate = lambda i: patch.set_data(deep_frames[i])
# ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(deep_frames), interval = 50)
# HTML(ani.to_jshtml())

"""### Generate Expert Trajectories (TODO):

Using the Q-learning agent above, please complete this block of code to generate expert trajectories

#### Get actions from expert for a specific observation
"""

def get_expert_action(Q,discretization,env,state):
  '''
  TODO: Implement this function

  NOTE: YOU WILL BE USING THIS FUNCTION FOR THE IMITATION LEARNING SECTION AS WELL 

  Functionality: For a given state, returns the action that the expert would take 

  Input: Q value , numpy array
         the discretization 
         env , the environment
         state, (Box(2,)): the observation space state, here it is [position, velocity]

  Returns: action, has dimensions of the action space and type of action space
  '''
  state_disc = discretize(state,discretization,env)
  action = np.argmax(Q[state_disc[0], state_disc[1]])

  return action

"""#### Generate Expert Trajectory"""

def generate_expert_trajectories(Q, discretization, env, num_episodes=150, data_path='./data'):

  '''
  TODO: Implement this function

  Functionality: Execute Expert Trajectories and Save them under the folder of data_path/

  Input: Q value , numpy array
         the discretization 
         env , the environment
         num_episodes, int, which is used to denote number of expert trajectories to store 
         
  Returns: total_samples, int, which denotes the total number of samples that were stored
  '''
  import os
  if not os.path.exists(data_path):
     os.makedirs(data_path)
  total_samples = 0

  for i in range(num_episodes):
    # Initial environment state
    state = env.reset()
    episode_observations = []
    episode_actions = []
    episode_dict = {'observations':[], 'actions':[]}
    done = False # indicates whether the episode is done
    terminal = False
    while done != True:                 
      # Get expert action from Q learner
      action = get_expert_action(Q,discretization,env,state)                                    
      # Get next_state, reward, and done using env.step()     
      next_state, reward, done, info = env.step(action) 
      # Update terminal
      terminal = done and next_state[0] >= 0.5
      episode_observations.append(state)
      episode_actions.append(action)
      state = next_state
      total_samples += 1

    episode_dict['observations'] = episode_observations
    episode_dict['actions'] = episode_actions  

    np.savez_compressed(data_path+'/episode_number'+'_'+str(i)+'.npz',**episode_dict) #where i can be the episode number that you save 
    # total_samples += 1

  # episode_dict['observations'] = episode_observations
  # episode_dict['actions'] = episode_actions
  
  # import os
  # if not os.path.exists(data_path):
  #    os.makedirs(data_path)
  # np.savez_compressed(data_path+'/episode_number'+'_'+str(i)+'.npz',**episode_dict) #where i can be the episode number that you save
  
  # print(num_episodes)
  # print(total_samples)

  return total_samples

"""#### Launch code for generating trajectories"""

# num_episodes = 100
# data_path = './data'


# total_samples = generate_expert_trajectories(Q,discretization,env,num_episodes,data_path) ## Generate trajectories. Use Q, discretization and env by running the previous section

# print('--------- Total Samples Recorded were --------', total_samples)

"""## Imitation Learning

Using the trajectories that you collected from the expert above, you will work on imitation learning agents in the code sections below

### Working with Data (TODO)

#### Loading Initial Expert Data
"""

# def get_args():
#   class Args(object):
#     pass

#   args = Args();
#   args.datapath = './data/'
#   args.episodes = 100
#   args.env = env
#   args.Q = np.load('./expert_Q.npy',allow_pickle=True)
#   args.discretization = np.array([10,100])

#   return args

def load_initial_data(args):
  '''
  TODO: Fill this function

  Functionality: Reads data from directory and converts them into numpy arrays of observations and actions

  Input arguments: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
  
  Returns: training_observations: numpy array, of shape (B,dim_of_observation), where B is total number of samples that you select
           training_actions: numpy array, of shape (B,dim_of_action), where B is total number of samples that you select

  '''
  training_observations = []
  training_actions = []
  num_of_episodes = 0

  for filename in os.listdir(args.datapath):
    if num_of_episodes >= args.initial_episodes_to_use:
      break
    episode_dict = np.load(args.datapath+'/'+filename)
    training_observations.append(episode_dict['observations'])
    training_actions.append(episode_dict['actions'])
    num_of_episodes += 1

  training_observations = np.concatenate(training_observations, axis=0)
  training_actions = np.concatenate(training_actions, axis=0)

  return training_observations, training_actions

# training_observations, training_actions = load_initial_data(args)
# # print(training_observations.shape)
# # print(training_actions.shape)

"""#### Convert numpy arrays to a Dataloader"""

def load_dataset(args, observations, actions, batch_size=64, data_transforms=None, num_workers=0):
  '''
  TODO: Fill this function fully 

  Functionality: Converts numpy arrays to dataloaders. 
  
  Inputs: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
          observations, numpy array, of shape (B,dim_of_observation), where B is number of samples 
          actions, numpy array, of shape (B,dim_of_action), where B is number of samples 
          batch_size, int, which you can play around with, but is set to 64 by default. 
          data_transforms, whatever transformations you want to make to your data.

  Returns: dataloader  
  '''
  from torch.utils import data
  dataset = data.TensorDataset(torch.FloatTensor(observations), torch.LongTensor(actions))
  dataloader = data.DataLoader(dataset,batch_size=batch_size)

  return dataloader

# dataloader = load_dataset(args, training_observations, training_actions)

"""#### Process Individual Observations"""

def process_individual_observation(args,observation):
  '''
  TODO: Fill this function fully 

  Functionality: Converts individual observations according to the pre-processing that you want  
  
  Inputs: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
          observations, shape (dim_of_observation)

  Returns: data, processed observation that can be fed into the model
  '''
  data = torch.FloatTensor(observation)

  return data

"""### Defining Networks (TODO)

#### Define your network for working from States
"""

class StatesNetwork(nn.Module):
    '''
    TODO: Implement this class
    '''
    def __init__(self, env):
        """
        Your code here
        """
        super(StatesNetwork, self).__init__()
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden1 = 256
        self.hidden2 = 128
        # print('Hidden: ',self.hidden)

        self.fc1 = nn.Linear(self.observation_space, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.hidden1, bias=False)
        self.fc3 = nn.Linear(self.hidden1, self.hidden2, bias=False)
        self.fc4 = nn.Linear(self.hidden2, self.hidden2, bias=False)
        self.fc5 = nn.Linear(self.hidden2, self.action_space, bias=False)
    
    def forward(self, x):    
        """
        Your code here
        @x: torch.Tensor((B,dim_of_observation))
        @return: torch.Tensor((B,dim_of_actions))
        """
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        
        # # forward_pass = x.max(0)[1]
        # # x = torch.argmax(x).item()
        # # F.softmax(x, dim=1) #
        # forward_pass = x

        # return forward_pass
        model = torch.nn.Sequential(self.fc1,self.fc2,self.fc3,self.fc4,self.fc5)
        return model(x)

class StatesNetwork2(nn.Module):
    '''
    TODO: Implement this class
    '''
    def __init__(self, env):
        """
        Your code here
        """
        super(StatesNetwork, self).__init__()
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 5
        print('Hidden: ',self.hidden)

        self.fc1 = nn.Linear(self.observation_space, self.hidden, bias=False)
        # self.fc2 = nn.Linear(self.hidden, self.hidden, bias=False)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.hidden, self.action_space, bias=False)
    
    def forward(self, x):    
        """
        Your code here
        @x: torch.Tensor((B,dim_of_observation))
        @return: torch.Tensor((B,dim_of_actions))
        """
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        
        # # forward_pass = x.max(0)[1]
        # # x = torch.argmax(x).item()
        # # F.softmax(x, dim=1) #
        # forward_pass = x

        # return forward_pass
        model = torch.nn.Sequential(self.fc1,self.dropout,self.fc2)
        return model(x)

"""### Training the model (TODO)"""

def train_model(args):

    '''
    TODO: Fill in the entire train function

    Functionality: Trains the model. How you train this is upto you. 

    Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 

    Returns: The trained model 

    '''
    args.env.seed(42)
    np.random.seed(42)
    # training_observations, training_actions = load_initial_data(args)
    # train_dataloader = load_dataset(args, training_observations, training_actions)
    train_dataloader = load_dataset(args, args.observations, args.actions)
    model = args.model
    # loss = nn.MSELoss()
    # loss = nn.NLLLoss()
    loss = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.075)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for i in range(args.num_epochs):

      # Enumeration loop of dataloader
      for idx, (X,y) in enumerate(train_dataloader):
        # Zero out gradients of optimizer
        optimizer.zero_grad()
        # Input training into model
        y_pred = model.forward(X)
        # y_pred = predict(model,X)
        # print(X.data)
        # print(y_pred.data)
        # print(y.data)
        # Calculate loss
        # train_loss = loss(y_pred,y.type(torch.LongTensor))
        train_loss = loss(y_pred,y)
        # if idx % 20 == 19:
        #   print(train_loss.item())
        # Backpropagate
        train_loss.backward()
        # Step the optimizer
        optimizer.step()
    
    return model

"""### DAgger (TODO)

#### Get the expert trajectory for imitating agent's observations
"""

def execute_dagger(args):

  '''
  TODO: Implement this function

  Functionality: Collect expert labels for the observations seen by the imitation learning agent 
  
  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
         
  Returns: imitation_observations, a numpy array that has dimensions of (episode_length,dim_of_observation)
           expert_actions, a numpy array that has dimensions of (episode_length,dim_of_action)
  '''
  env = args.env
  state = env.reset()
  Q = args.Q
  discretization = args.discretization
  imitation_observations = []
  expert_actions = []
  done = False # indicates whether the episode is done
  terminal = False

  while done != True:                 
    # Get expert action from Q learner
    action = get_expert_action(Q,discretization,env,state)                                    
    # Get next_state, reward, and done using env.step()     
    next_state, reward, done, info = env.step(action) 
    # Update terminal
    terminal = done and next_state[0] >= 0.5
    imitation_observations.append(state)
    expert_actions.append(action)
    state = next_state
  
  imitation_observations = np.array(imitation_observations)
  expert_actions = np.array(expert_actions)
          
  return imitation_observations, expert_actions

"""#### Aggregate new rollout to the full dataset"""

def aggregate_dataset(training_observations, training_actions, imitation_states, expert_actions):

  '''
  TODO: Implement this function

  Functionality: Adds new expert labeled rollout to the overall dataset

  Input: training_observations, a numpy array that has dimensions of (dataset_size,dim_of_observation)
         training_actions, a numpy array that has dimensions of (dataset_size,dim_of_action)
         imitation_observations, a numpy array that has dimensions of (episode_length,dim_of_observation)
         expert_actions, a numpy array that has dimensions of (episode_length,dim_of_action)

  Returns: training_observations, a numpy array that has dimensions of (updated_dataset_size,dim_of_observation)
           training_actions, a numpy array that has dimensions of (updated_dataset_size,dim_of_action)
  '''
  training_observations = np.concatenate((training_observations,imitation_states), axis=0)
  training_actions = np.concatenate((training_actions,expert_actions), axis=0)

  return training_observations, training_actions

"""### Utility

#### Code for prediction of the network and calculating the accuracy
"""

# import numpy as np
# from torchvision.transforms import functional as TF

# def accuracy(outputs, labels):
#     outputs_idx = outputs.max(1)[1].type_as(labels)
#     return outputs_idx.eq(labels).float().mean()

# def predict(model, inputs, device='cpu'):
#     inputs = inputs.to(device)
#     logits = model(inputs)
#     return F.softmax(logits, -1)

"""#### Wrapper for Rendering the environment 

Same code that was used in the Q-learning agent
"""

class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):

        from PIL import Image
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)

"""### Test model performance"""

def test_model(args, record_frames=False):

    '''
  Functionality: Should take your model and run it for a complete episode (model should either not finish the game in 200 steps or finish the game). Record stats

  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
         record_frames, Boolean. Denotes if you want to record frames to display them as video.

  Returns: final_position, The final position of the car when the episode ended
           success, Boolean, denotes if the episode was a success or not
           frames, a list of frames that have been rendered throughout the episode. Should have a length of the total episode length
           episode_reward, float, denotes the total rewards obtained while executing this episode
  '''

    frames = []
    env = args.env 
    model = args.model
    state = env.reset()

    model.eval()

    episode_reward = 0

    success = False
    done = False 

    while not done:

        observation = state

        data = process_individual_observation(args,observation)
        logit = model(data)
        action = torch.argmax(logit).item()

        if record_frames: #You can change the rate of recording frames as you like
            frames.append(env.render())

        next_state, reward, done, _ = env.step(action) 
        episode_reward += reward

        if done:    
            if next_state[0] >= 0.5:
                success = True
            final_position = next_state[0]
            return final_position,success, frames, episode_reward
        else:
            state = next_state

"""### Main Imitation Learning Method (TODO)"""

# def imitate(args):
#   '''
#   TODO: Implement this function

#   Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 

#   Functionality: For a given set of args, performs imitation learning as desired. 

#   Returns: final_positions, A list of final positions achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
#            success_history, A list of success percentage achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
#            reward_history, A list of episode rewards achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
#            frames, A list of video frames of the model executing its policy every time it is tested, can choose to not record. Should have a length of the number of times you chose to record frames
#            args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access
#   '''
#   final_positions = []
#   success_history = []
#   frames = []
#   reward_history = []

#   # args.observations, args.actions = load_initial_data(args)

#   # if args.do_dagger == False:
#   args.model = train_model(args)

#   final_position, success, episode_frames, episode_reward = test_model(args, record_frames=args.record_frames)
#   final_positions.append(final_position)
#   success_history.append(success)
#   frames.extend(episode_frames)
#   reward_history.append(episode_reward)
  
#   return final_positions, success_history, frames, reward_history, args

def imitate(args):
  '''
  TODO: Implement this function

  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 

  Functionality: For a given set of args, performs imitation learning as desired. 

  Returns: final_positions, A list of final positions achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
           success_history, A list of success percentage achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
           reward_history, A list of episode rewards achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
           frames, A list of video frames of the model executing its policy every time it is tested, can choose to not record. Should have a length of the number of times you chose to record frames
           args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access
  '''
  final_positions = []
  success_history = []
  frames = []
  reward_history = []

  args.observations, args.actions = load_initial_data(args)

  if args.do_dagger == True:
    for i in range(args.max_dagger_iterations):
      imitation_states, expert_actions = execute_dagger(args)
      args.observations, args.actions = aggregate_dataset(
          args.observations, args.actions, imitation_states, expert_actions)
      args.model = train_model(args)

  else:
    args.model = train_model(args)

  final_position, success, episode_frames, episode_reward = test_model(args, record_frames=args.record_frames)
  final_positions.append(final_position)
  success_history.append(success)
  frames.extend(episode_frames)
  reward_history.append(episode_reward)
  
  return final_positions, success_history, frames, reward_history, args

"""### Launch Imitation Learning (TODO)
Define Args and Launch 'imitate'
"""

# '''
# TODO: Expand the attributes of Args as you please. 

# But please maintain the ones given below, i.e: You should be using the ones given below. Fill them out.

# Some of these are already filled out for you. 
# '''


# ##TODO: Fill in the given attributes (you should use these in your code), and add to them as you please.
# def get_args():
#   class Args(object):
#     pass

#   args = Args();
#   args.datapath = './data/'
#   # args.env = env
#   args.resize_observation_shape = 100
#   args.env = gym.make('MountainCar-v0')
#   args.env = ResizeObservation(args.env, args.resize_observation_shape)

#   args.do_dagger = False
#   args.max_dagger_iterations = 18
#   # if not args.do_dagger:
#   #   assert args.max_dagger_iterations==1
#   args.record_frames = True
#   args.initial_episodes_to_use = 20
#   args.model = StatesNetwork(args.env)
#   args.num_epochs = 2
#   args.Q = np.load('./expert_Q.npy',allow_pickle=True)
#   args.discretization = np.array([10,100])
#   args.training_observations = None
#   args.training_actions = None


#   # positions, successes, frames, reward_history, args = imitate(args)
#   return args

'''
TODO: Expand the attributes of Args as you please. 

But please maintain the ones given below, i.e: You should be using the ones given below. Fill them out.

Some of these are already filled out for you. 
'''


##TODO: Fill in the given attributes (you should use these in your code), and add to them as you please.
def get_args():
  class Args(object):
    pass

  args = Args();
  args.datapath = './data/'
  # args.env = env
  args.resize_observation_shape = 100
  args.env = gym.make('MountainCar-v0')
  args.env = ResizeObservation(args.env, args.resize_observation_shape)

  args.do_dagger = False
  args.max_dagger_iterations = 18
  # if not args.do_dagger:
  #   assert args.max_dagger_iterations==1
  args.record_frames = True
  args.initial_episodes_to_use = 20
  if args.do_dagger == False:
    args.model = StatesNetwork(args.env)
  else:
    args.model = StatesNetwork2(args.env)
  args.num_epochs = 2
  args.Q = np.load('./expert_Q.npy',allow_pickle=True)
  args.discretization = np.array([10,100])
  args.training_observations = None
  args.training_actions = None


  # positions, successes, frames, reward_history, args = imitate(args)
  return args

"""### Average Performance Metrics

Use this function to see how well your agent is doing.
"""

def get_average_performance(args, run_for=1000):

  final_positions = 0
  successes = 0
  rewards = 0

  for ep in range(run_for):
    pos, success, _, episode_rewards = test_model(args, record_frames=False)   #test imitation policy
    final_positions += pos 
    rewards += episode_rewards
    if success:
      successes += 1
    print('Running Episode: ',ep,' Success: ', success)
    average_final_positions = final_positions/(ep+1)
    average_success_rate = 100*(successes/(ep+1))
    average_episode_rewards = rewards/(ep+1)

  print('Average Final Position achieved by the Agent: ',average_final_positions)
  print('Average Success Rate achieved by the Agent: ',average_success_rate)
  print('Average Episode Reward achieved by the Agent: ',average_episode_rewards)
  print('Episodes: ',args.initial_episodes_to_use)
  print('Epochs: ',args.num_epochs)

  return average_final_positions, average_success_rate, average_episode_rewards 


# final_pos, succ_rate, ep_rwds = get_average_performance(args)

# print('Average Final Position achieved by the Agent: ',final_pos)
# print('Average Success Rate achieved by the Agent: ',succ_rate)
# print('Average Episode Reward achieved by the Agent: ',ep_rwds)

"""### Visualization

#### Plotting code

Use the code below to make plots to see how well your agent did as it trained.
"""

# import pandas as pd 

# plt.plot(successes)
# plt.xlabel('Episode')
# plt.ylabel('% of Episodes with Success')
# plt.title('% Successes')
# plt.show()
# plt.close()

# p = pd.Series(positions)
# ma = p.rolling(3).mean()
# plt.plot(p, alpha=0.8)
# plt.plot(ma)
# plt.xlabel('Episode')
# plt.ylabel('Position')
# plt.title('Car Final Position')
# plt.show()

# plt.plot(reward_history)
# plt.xlabel('Episode')
# plt.ylabel('Episode Rewards Achieved')
# plt.title('Episode Rewards')
# plt.show()
# plt.close()

"""#### Make a video!

Using the frames that you recorded in ``` frames ```, Run the code below to display a video that you can use to see how well your agent is doing
"""

# #### Video plotting code #####################
# deep_frames = []
# for f in frames:
#   deep_frames += f
# plt.figure(figsize=(deep_frames[0].shape[1] / 72.0, deep_frames[0].shape[0] / 72.0), dpi = 72)
# patch = plt.imshow(deep_frames[0])
# plt.axis('off')
# animate = lambda i: patch.set_data(deep_frames[i])
# ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(deep_frames), interval = 50)
# HTML(ani.to_jshtml())
