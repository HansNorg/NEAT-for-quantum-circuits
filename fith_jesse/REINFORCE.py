""""This code closely follows the Tensorflow Quantum tutorial notebook
"Parametrized Quantum Circuits for Reinforcement Learning". Credits therefore are due to the developers of Tensorflow Quantum
"""

import tensorflow as tf
# import tensorflow_quantum as tfq

# import cirq, sympy
# import gymnasium as gym
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
# from cirq.contrib.svg import SVGCircuit

from fox_in_a_hole_gym import FoxInAHole, FoxInAHolev2, FoxInAHoleBounded, QFIAHv1, QFIAHv2

class reinforce_agent():
  def __init__(self, batch_size):
    self.batch_size=batch_size
    
    pass

  def gather_episodes(self,state_bounds, n_holes, n_actions, model, batch_size, env_name, len_state, max_steps, prob_1, prob_2):
      """Interact with environment in batched fashion."""

      trajectories = [defaultdict(list) for _ in range(batch_size)]
    
      if env_name.lower() =="foxinahole":
         envs = [FoxInAHole(n_holes=n_holes) for _ in range(batch_size)]
        
      elif env_name.lower()=="foxinaholev2":
         envs = [FoxInAHolev2(n_holes=n_holes, max_steps = max_steps, len_state=len_state) for _ in range(batch_size)]

      elif env_name.lower()=="foxinaholebounded":
         envs = [FoxInAHoleBounded(n_holes=n_holes) for _ in range(batch_size)]

      elif env_name.lower() == "qfiahv1":
         envs = [QFIAHv1(n_holes, len_state, max_steps, prob_1, prob_2) for _ in range(batch_size)]
      
      elif env_name.lower()=="qfiahv2":
         envs = [QFIAHv2(n_holes, len_state, max_steps, prob_1, prob_2, tunneling_prob= 0.2) for _ in range(batch_size)]

      else:
         raise KeyError

      done = [False for _ in range(batch_size)]
      states = [e.reset() for e in envs]

      while not all(done):
          unfinished_ids = [i for i in range(batch_size) if not done[i]]
          normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]


          for i, state in zip(unfinished_ids, normalized_states):
              trajectories[i]['states'].append(state)

          # Compute policy for all unfinished envs in parallel
          action_probs = model(states)
          # Store action and transition all environments to the next state
          states = [None for i in range(batch_size)]
          for i, policy in zip(unfinished_ids, action_probs.numpy()):
              action = np.random.choice(n_actions, p=policy)
              states[i], reward, done[i], _ = envs[i].step(action) 
              trajectories[i]['actions'].append(action)
              trajectories[i]['rewards'].append(reward)

      return trajectories

  def compute_returns(self, rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()

    return returns


