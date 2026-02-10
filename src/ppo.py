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
        # Lists to store raw data from the environment
        self.states = []       # Graph Data objects (or feature tensors)
        self.actions = []      # Action indices
        self.rewards = []      # Raw rewards (+0.1, -1.0, etc.)
        self.dones = []        # Boolean flags
        self.log_probs = []    # Log probabilities from the OLD policy
        self.values = []       # Critic predictions from the OLD policy
        
        # Lists to store processed data for training
        self.advantages = []
        self.returns = []

    def push(self, state_data, action, reward, done, log_prob, value):
        """Save one step of interaction."""
        # For Graphs, we might store the whole Data object or just features.
        # Here we store the tuple (x, edge_index, batch) for simplicity
        self.states.append(state_data) 
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def calculate_gae(self, last_value):
        """
        Phase 2: Process the buffer to calculate GAE and Returns.
        last_value: The Critic's prediction for the state AFTER the final step.
        """
        advantages = []
        gae = 0
        
        # Append the "next value" to the end of the list to simplify the loop
        values = self.values + [last_value]
        
        # Iterate backwards (from last step to first)
        for i in reversed(range(len(self.rewards))):
            # If done=True, the "next state" is terminal, so value is 0.
            mask = 1 - self.dones[i] 
            
            # Delta = Reward + gamma * Next_Value - Current_Value
            delta = self.rewards[i] + self.gamma * values[i+1] * mask - values[i]
            
            # GAE = Delta + gamma * lambda * Next_GAE
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            
            # Insert at the front (since we are iterating backwards)
            advantages.insert(0, gae)
        
        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Returns = Advantage + Value (Standard PPO practice)
        # This gives the target for the Critic
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)

    def get_batches(self, batch_size):
        """
        Yields mini-batches of tensors for training.
        """
        total_samples = len(self.rewards)
        indices = torch.randperm(total_samples) # Shuffle indices
        
        for start_idx in range(0, total_samples, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            
            # Gather batch data, custom logic for the graph batches ('states' is a list of tuples (x, edge_index, batch)) is in the main loop
            # Indices return so the main loop can slice the state list.
            yield idx, \
                  self.advantages[idx], \
                  self.returns[idx], \
                  torch.tensor(self.actions)[idx].to(self.device), \
                  torch.tensor(self.log_probs)[idx].to(self.device)