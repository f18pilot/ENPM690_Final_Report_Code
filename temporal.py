#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:20:25 2020

This program uses the temporal difference method to calculate a ValueMap 
and convergence graphs and determine robot's path to goal. This program is 
based on Chapter 5 of Sutton and Barto's 'ReinforcementLearning', and also 
on the online articles modeling gridworld from Zhang and Martinez:
https://towardsdatascience.com/reinforcement-learning-rl-101-with-python
https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff

@author: Brenda
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
#%pylab inline
import random

# parameters
gamma = 0.1   # discounting rate
rewardAmount = -1
rewardAmount_obstacle = -10
grid = 10
alpha = 0.1 
terminalStates = [[0,7]]   
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
Iterations = 5000
obstacleStates = []  #[4,5], [8,3]
initialState = [9,9]
# initialization
V = np.zeros((grid, grid))
for obstacle in obstacleStates:
    #print(obstacle)
    #print(obstacle[0])
    V[obstacle[0], obstacle[1]] += V[obstacle[0], obstacle[1] ] + rewardAmount_obstacle
    V[initialState[0], initialState[1]] += V[initialState[0], initialState[1] ] + rewardAmount_obstacle  
returns = {(i, j):list() for i in range(grid) for j in range(grid)}
deltas = {(i, j):list() for i in range(grid) for j in range(grid)}
states = [[i, j] for i in range(grid) for j in range(grid)]


def generateNextAction():
    return random.choice(actions)

def takeAction(state, action):
    if list(state) in terminalStates:
        return 0, None
    finalState = np.array(state)+np.array(action)
    reward = rewardAmount
    fs = list(finalState)
    # if robot crosses wall
    if -1 in list(finalState) or grid in list(finalState):
        finalState = state
        reward = rewardAmount
    if fs in obstacleStates:
        finalState = state
        reward = rewardAmount_obstacle
            
    return reward, list(finalState)

for it in tqdm(range(Iterations)):
    state = initialState
    while True:
        action = generateNextAction()
        reward, finalState = takeAction(state, action)
        
        # we reached the end
        if finalState is None:
            break
    
        # modify Value function
        before =  V[state[0], state[1]]
        V[state[0], state[1]] += alpha*(reward + gamma*V[finalState[0], finalState[1]] - V[state[0], state[1]])
        deltas[state[0], state[1]].append(float(np.abs(before-V[state[0], state[1]])))
        
        state = finalState
    if it in [0,1, Iterations-1]:
            print("Iteration {}".format(it+1))
            
            print(V)
            print("")  

plt.figure()
all_series = [list(epi)[:50] for epi in deltas.values()]
for series in (all_series):
    plt.plot(series)
