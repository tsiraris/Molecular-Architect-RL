import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PPOBuffer(Dataset):
    """
    Stores training data (trajectories) and calculates advantages (GAE).
    Acts as a PyTorch Dataset for easy batching.
    """
    def __init__(self, device, gamma=0.99, gae_lambda=0.95):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        """
        Clears all stored data. Call at the beginning of each PPO update cycle.
        """
        # Lists to store raw data from the environment
        self.states = []       # To store complex tuples (x, edge_index, edge_attr...)
        self.actions = []      # Action indices
        self.rewards = []      # Raw rewards
        self.dones = []        # Boolean flags
        self.log_probs = []    # Log probabilities from the OLD policy
        self.values = []       # Critic predictions from the OLD policy
        
        # Lists to store processed data for training
        self.advantages = []
        self.returns = []

    def push(self, state_data, action, reward, done, log_prob, value):
        """
        Save one step of interaction.
        """
        self.states.append(state_data) 
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def calculate_gae(self, last_value):
        """
        Phase 2: Process the buffer to calculate GAE and Returns.
        """
        advantages = []
        gae = 0
        
        # Append the "next value" to simplify the loop
        values = self.values + [last_value]
        
        for i in reversed(range(len(self.rewards))):
            mask = 1 - self.dones[i] 
            delta = self.rewards[i] + self.gamma * values[i+1] * mask - values[i]    # TD error (mask closes the episode if done)
            gae = delta + self.gamma * self.gae_lambda * mask * gae                  # GAE recursion 
            advantages.insert(0, gae)
        
        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)

    def get_batches(self, batch_size):
        """
        Yields mini-batches of tensors for training.
        """
        total_samples = len(self.rewards)
        indices = torch.randperm(total_samples) 
        
        for start_idx in range(0, total_samples, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            
            yield idx, \
                  self.advantages[idx], \
                  self.returns[idx], \
                  torch.tensor(self.actions)[idx].to(self.device), \
                  torch.tensor(self.log_probs)[idx].to(self.device)