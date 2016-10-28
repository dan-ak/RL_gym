# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:42:16 2016

@author: dan
"""

import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, gamma=1.0, theta=0.00001):
    
    V = np.zeros(env.nS)
    
    while True:
        max_err = 0
        
        for s in range(env.nS):
            v = 0
            for a, a_prob in enumerate(policy[s]):
               for  prob, next_state, reward, done in env.P[s][a]:
                  v += a_prob * prob * (reward + gamma*V[next_state])
                  
            max_err = max(max_err, np.abs(V[s]-v))
            
            V[s] = v
        
        
        if max_err < theta:
            break
    
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, gamma=1.0):

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval_fn(policy, env)
    
    while True:
           
        
        new_policy = np.zeros([env.nS, env.nA])
        
        for s in range(env.nS):
            
            a_v = np.zeros(env.nA)
            
            for a in range(env.nA):
                for  prob, next_state, reward, done in env.P[s][a]:
                    a_v[a] += prob * (reward + gamma * v[next_state])
            
            new_policy[s][np.argmax(a_v)] = 1.0
            
        new_v = policy_eval_fn(new_policy, env)    
            
        if np.allclose(v, new_v, atol = 1e-5):    
            break
    
        policy = new_policy
        v = new_v
    
    return policy, v


def value_iteration(env, theta=0.0001, gamma=1.0):
    
    v = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    max_err = 0
    
    while True:
        
        max_err = 0
        
        for s in range(env.nS):
            
            a_v = np.zeros(env.nA)
            
            for a in range(env.nA):
                for  prob, next_state, reward, done in env.P[s][a]:
                    a_v[a] += prob * (reward + gamma * v[next_state])
        
            new_a = np.argmax(a_v)

            max_err = max(max_err, np.abs(v[s]-a_v[new_a]))

            policy[s] = np.eye(env.nA)[new_a]
            v[s] = a_v[new_a]

        
        if max_err < theta:
            break
        
    return policy, v

    
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

policy, v = policy_improvement(env)
policy, v = value_iteration(env)

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)