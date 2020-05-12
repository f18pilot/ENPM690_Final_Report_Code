#!/usr/bin/env python3
# -*- coding: utf-8 -*-https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff
"""
Created on Mon Apr 13 21:18:04 2020
RL in gridworld

This code computes the Value function for the 10 x 10 grid playing field for
the Pirate robot, and then computes the robot's path.  This code was written 
based on Chapters 3 &  4 from Reinforcement Learning by Sutton and Barto, and
from online articles by George Martinez 
https://towardsdatascience.com/reinforcement-learning-rl-101-with-python
and Jeremy Zhang 
https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff


@author: Brenda
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

rewardAmount = -1
rewardAmount_obstacle = -2
grid = 10
gamma = 0.9 # discounting rate
Iterations = 1000
terminalStates =  [[0,7]]  #object or Goal Zone position
obstacleStates = [[4,5], [8,3]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
initialState = [9,9]
print(initialState)

def defineReward(initialPosition, action):
    if initialPosition in terminalStates:
        return initialPosition, 0
    
    finalPosition = np.array(initialPosition) + np.array(action)    
    reward = rewardAmount
    fp = list(finalPosition)

    if -1 in finalPosition or grid in finalPosition:    
        finalPosition = initialPosition
        return initialPosition, reward
    
    if fp in obstacleStates:
        finalPosition = initialPosition
        return finalPosition, rewardAmount_obstacle
    if fp in initialState:
        return initialPosition, 0

    return finalPosition, reward


# Build the initial ValueMap
valueMap = np.zeros((grid, grid))

# Add in obstacles
for obstacle in obstacleStates:
   
    valueMap[obstacle[0], obstacle[1]] += valueMap[obstacle[0], obstacle[1] ] + rewardAmount_obstacle
valueMap[initialState[0], initialState[1]] += valueMap[initialState[0], initialState[1] ] + rewardAmount_obstacle
states = [[i, j] for i in range(grid) for j in range(grid)]
 
states.insert(0, states.pop(states.index(initialState)))
 
# values of the value function at step 0
print(valueMap)

deltas = []
for it in tqdm(range(Iterations)):

    ValueMap = np.copy(valueMap)
    delta = []
    for state in states:
        V = 0
        for action in actions:
            finalPosition, reward = defineReward(state, action)
            V += (0.25)*(reward+(gamma*valueMap[finalPosition[0], finalPosition[1]]))
        # eliminate obstacles which grow every iteration
    
        #if np.abs(ValueMap[state[0], state[1]]-V)  < 100:  
            delta.append(np.abs(ValueMap[state[0], state[1]]-V) )
            ValueMap[state[0], state[1]] = V
        
    deltas.append(delta)
    # take account of the obstacles
    for obstacle in obstacleStates:
        ValueMap[obstacle[0], obstacle[1]] += ValueMap[obstacle[0], obstacle[1] ] + rewardAmount_obstacle
    valueMap = ValueMap
    if it in [0,1,2,Iterations-1]:    
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")
'''
Robot actions are inputted to the control algorithm for the Pirate and converted to commands based on 
robot heading. For robot initial heading (assumed 0 deg), [-1, 0] is forward
[0,-1] is to the right, for robot heading 270, forward is [0,1], etc  Any heading change,
i.e., a right or left motion, commands a 90 degree turn
'''
# Compute action sequence from ValueMap
value0 = -10000
robotActions = []
terminalState = list(terminalStates)
rStates = []
while initialState[1] != terminalState[0][1] or initialState[0] != terminalState[0][0]:

    for action in actions:
            newState =  np.add(initialState, action)
            if -1 in newState or grid in newState:
                    #print("out of bounds", newState)
                    pass
            elif valueMap[newState[0], newState[1]] > value0:
                    
                    actionFinal = action
                    maxState = newState
                    value0 = valueMap[newState[0], newState[1]]
                    rStates.append(list(maxState))
    
   
    initialState = maxState
    robotActions.append(actionFinal)
     
    

print('robotActions', robotActions)
   