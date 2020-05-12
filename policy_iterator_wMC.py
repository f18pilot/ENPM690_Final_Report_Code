#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:32:14 2020

This program uses Monte Carlo methods to calculate ValueMap to determine robot
path.  This program is based on Chapter 5 of Sutton and Barto's 'Reinforcement
Learning', and also the online articles of Zhang and Martinez:
https://towardsdatascience.com/reinforcement-learning-rl-101-with-python
https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff

@author: Brenda
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# parameters
gamma = 1 # discounting rate
rewardAmount = -1
rewardAmount_obstacle = -10
grid = 10
terminalStates = [[0,7]]  
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
obstacleStates = [[4,5], [8,3]]
Iterations = 500
initialState = [9,9]


# create V(s)
V = np.zeros((grid, grid))
for obstacle in obstacleStates:
    #print(obstacle)
    #print(obstacle[0])
    V[obstacle[0], obstacle[1]] += V[obstacle[0], obstacle[1] ] + rewardAmount_obstacle
    V[initialState[0], initialState[1]] += V[initialState[0], initialState[1] ] + rewardAmount_obstacle   
returns = {(i, j):list() for i in range(grid) for j in range(grid)}
deltas = {(i, j):list() for i in range(grid) for j in range(grid)}
states = [[i, j] for i in range(grid) for j in range(grid)]


def generateEpisode():
    initialState = [9,9]   #pick our start point
               
    episode = []
    while True:
        if list(initialState) in terminalStates:
            return episode
        action = random.choice(actions)
        finalState = np.array(initialState)+np.array(action)
        fs = list(finalState)
        if -1 in list(finalState) or grid in list(finalState):
            finalState = initialState
        if fs in obstacleStates:
            finalState = initialState
            episode.append([list(initialState), action, rewardAmount_obstacle, list(initialState)])
        else:
            episode.append([list(initialState), action, rewardAmount, list(finalState)])
            initialState = finalState
       
for it in tqdm(range(Iterations)):
    episode = generateEpisode()
    G = 0
    #print(episode)
    for i, step in enumerate(episode[::-1]):
         
        G = gamma*G + step[2]  # Adding up rewards
        if step[0] not in [epi[0] for epi in episode[::-1][len(episode)-i:]]:
            i = (step[0][0], step[0][1])
            returns[i].append(G)
            newValue = np.average(returns[i])
            deltas[i[0], i[1]].append(np.abs(V[i[0], i[1]]-newValue))
            V[i[0], i[1]] = newValue
            # account for obstacles not being visited
            for obstacle in obstacleStates:
                V[obstacle[0], obstacle[1]] +=  0.0001*rewardAmount_obstacle      
            
plt.figure(figsize=(20,10))
#plot only the last x step averages values (otherwise the deltas are too large)
all_series = [list(epi)for epi in deltas.values()] #[:40]
for series in all_series:
    plt.plot(series)
plt.show()