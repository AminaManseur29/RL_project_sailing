"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from agents.base_agent import BaseAgent

class QLearningTrainedAgent(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # State discretization parameters
        self.position_bins = 8
        self.velocity_bins = 4
        self.wind_bins = 8
        
        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(4, 0, 1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 0, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 3, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 3, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 3, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 3, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 1, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 0, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 0, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 2, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 2, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 1, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 0, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 2, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 1, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 4, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 4, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 5, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 6, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 6, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 6, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 0, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 0, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 2, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 4, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 2, 4, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 4, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 4, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 5, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 6, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 6, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 6, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 0, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 7, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 0, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 1, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 0, 1, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 1, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 1, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 0, 1, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 1, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 2, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 0, 2, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 0, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 2, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 3, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 3, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 4, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 5, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 0, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 3, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 3, 1, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 4, 2, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 2, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 2, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 1, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 4, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 2, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 6, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 6, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 1, 7, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 3, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 7, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 0, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 2, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 2, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 1, 0, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 0, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 2, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 1, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 5, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 5, 0, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 5, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 6, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 6, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 6, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 7, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 2, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 3, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 0, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 0, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 1, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 1, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 1, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 2, 2, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 2, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 3, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 2, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 2, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 3, 0, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 4, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 4, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 4, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 4, 0, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 5, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 5, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 0, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 3, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 0, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 2, 1, 1, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 2, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 2, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 0, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 1, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 2, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 2, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 0, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 3, 2, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 1, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 2, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 2, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 3, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 3, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 3, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 0, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 0, 3, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 2, 2, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 1, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 3, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 3, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 0, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 2, 2, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 2, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 0, 3, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 6, 3, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 1, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 3, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 2, 3, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 3, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 0, 4, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 4, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 1, 4, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 2, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 3, 5, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 5, 6)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 5, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 3, 6, 7)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract position, velocity and wind from observation
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # Discretize position (assume 32x32 grid)
        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # Discretize velocity direction
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1) % self.velocity_bins
        
        # Discretize wind direction
        wind_direction = np.arctan2(wy, wx)  # Range: [-pi, pi]
        wind_bin = int(((wind_direction + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins
        
        # Return discrete state tuple
        return (x_bin, y_bin, v_bin, wind_bin)
        
    def act(self, observation):
        """Choose the best action according to the learned Q-table."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Use default actions if state not in Q-table
        if state not in self.q_table:
            return 0  # Default to North if state not seen during training
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def reset(self):
        """Reset the agent for a new episode."""
        pass  # Nothing to reset
        
    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)
