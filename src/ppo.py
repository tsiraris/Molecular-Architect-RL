import torch
from torch.utils.data import Dataset

class PPOBuffer(Dataset):
    """
    Experience Replay Buffer for PPO.
    Stores trajectories and computes Generalized Advantage Estimation (GAE).
    """
    def __init__(self, device, gamma=0.99, gae_lambda=0.95):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        """Reset buffer for the next update cycle."""
        # Raw storage (CPU RAM efficient lists)
        self.states = []       
        self.actions = []      
        self.rewards = []      
        self.dones = []        
        self.log_probs = []    
        self.values = []       
        
        # Computed tensors
        self.advantages = []
        self.returns = []

    def push(self, state_data, action, reward, done, log_prob, value):
        """Add a transition to the buffer."""
        self.states.append(state_data) 
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def calculate_gae(self, last_value):
        """
        Compute GAE and Returns.
        This smooths the reward signal and reduces variance in training.
        """
        advantages = []
        gae = 0
        
        values = self.values + [last_value]
        
        for i in reversed(range(len(self.rewards))):
            mask = 1 - self.dones[i] 
            delta = self.rewards[i] + self.gamma * values[i+1] * mask - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae 
            advantages.insert(0, gae)
        
        self.advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)

    def get_batches(self, batch_size):
        """
        Yields randomized mini-batches for PPO updates.
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