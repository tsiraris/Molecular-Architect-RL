"""
==========================================
Proximal Policy Optimization (PPO) Buffer
==========================================

This script defines the data structures necessary to collect, store, and process 
reinforcement learning rollouts across multiple parallel (vectorized) environments. 
It acts as the episodic memory for the Actor-Critic agent during training.

How it works:
1. As the agent interacts with `num_envs` parallel environments, each discrete step 
   is saved as a `Transition` and pushed to the `PPOBuffer`.
2. Trajectories are stored independently for each environment to prevent mixing 
   timesteps from different Markov Decision Processes (MDPs).
3. At the end of a rollout horizon, `finalize()` is called. It uses the Critic's 
   final value estimate to bootstrap and compute Generalized Advantage Estimation (GAE).
4. GAE balances bias and variance when estimating how much better an action was 
   compared to the baseline.
5. The environment-specific trajectories are then flattened into contiguous 1D PyTorch 
   tensors, and `get_batches()` yields randomized mini-batches for the PPO update epoch.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple
import numpy as np
import torch

# -----------------------------------------------------------------------------------------
# Transition Data Structure
# A lightweight container to hold the necessary components of a single MDP timestep.
# -----------------------------------------------------------------------------------------
@dataclass                                                                                  # Dataclass Decorator to auto-generate __init__, __repr__, __eq__ etc.
class Transition:
    """
    A single transition (step) in the environment.
    
    This dataclass encapsulates the state, chosen action, environment reward, termination 
    flag, policy log probability, and critic value estimate for a single moment in time. 
    It leverages Python's `@dataclass` decorator to automatically generate boilerplate 
    initialization and representation methods.
    
    Args:
        state (object): The observation/state at the given timestep (e.g., PyG graph tensors).
        action (int): The discrete integer action selected by the policy.
        reward (float): The scalar reward returned by the environment.
        done (float): A binary flag (1.0 or 0.0) indicating if the episode terminated.
        log_prob (float): The log probability of the action under the acting policy.
        value (float): The expected future return estimated by the Critic network.
        
    Example:
        >>> t = Transition(state="graph_data", action=5, reward=1.0, done=0.0, log_prob=-0.5, value=0.8)
        >>> print(t.reward)
        1.0
    """
    state: object                                                                           # opaque (graph tensors etc.) - Stores the environment observation for the current step
    action: int                                                                             # Stores the discrete integer action chosen by the actor network
    reward: float                                                                           # Stores the scalar float reward granted by the environment after the action
    done: float                                                                             # Stores a float (0.0 or 1.0) indicating if the episode ended at this step
    log_prob: float                                                                         # Stores the log-probability of taking this action, needed for PPO ratio calculation
    value: float                                                                            # Stores the Critic's scalar estimation of the state's value


class PPOBuffer:
    """
    Vectorized-environment PPO buffer with per-env GAE.
    
    Maintains independent lists of `Transition` objects for multiple parallel environments. 
    After collecting experiences for a set number of steps, it computes the 
    Generalized Advantage Estimation (GAE) per environment by iterating backwards through 
    the collected trajectory. Finally, it flattens the parallel trajectories into global 
    tensors to serve randomized mini-batches during PPO gradient descent.
    """
    def __init__(self, device: torch.device, num_envs: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initializes the PPO Buffer configuration and allocates internal storage structures.
        
        Stores the mathematical hyperparameters (discount factor and GAE lambda) and 
        compute device. It immediately calls `clear()` to initialize the empty lists 
        and null tensors that will hold rollout data.
        
        Args:
            device (torch.device): Compute device (CPU/GPU) where finalized tensors will reside.
            num_envs (int): The number of parallel vectorized environments collecting data.
            gamma (float, optional): The discount factor for future rewards. Defaults to 0.99.
            gae_lambda (float, optional): The GAE smoothing parameter (0 to 1). Defaults to 0.95.
            
        Returns:
            None.
            
        Example:
            >>> buffer = PPOBuffer(torch.device("cpu"), num_envs=4)
            >>> buffer.num_envs
            4
        """
        # ---------------------------------------------------------------------------------
        # Buffer Initialization
        # Bind device and hyperparameters, then setup the empty internal arrays.
        # ---------------------------------------------------------------------------------
        self.device = device                                                                # Store the intended PyTorch compute device for final tensor allocation
        self.num_envs = num_envs                                                            # Store the integer count of parallel environments being managed
        self.gamma = gamma                                                                  # Store the discount factor (gamma) for calculating present value of future rewards
        self.gae_lambda = gae_lambda                                                        # Store the lambda parameter controlling the bias-variance tradeoff in GAE
        self.clear()                                                                        # Execute the clearing routine to build the initial empty storage lists

    def clear(self) -> None:
        """
        Clears the buffer, wiping all previous trajectories and resetting tensors to None.
        
        Overwrites the trajectory storage with a list of empty lists (one per environment). 
        Resets the flattened state list to empty, and sets all finalized PyTorch tensors 
        to None. This prepares the buffer for a fresh rollout collection phase.
        
        Args:
            None.
            
        Returns:
            None.
            
        Example:
            >>> buffer = PPOBuffer(torch.device("cpu"), 2)
            >>> buffer.clear()
            >>> len(buffer._traj)
            2
        """
        # ---------------------------------------------------------------------------------
        # State Clearing
        # Wipe all cached trajectory data to prepare for the next rollout horizon.
        # ---------------------------------------------------------------------------------
        self._traj: List[List[Transition]] = [[] for _ in range(self.num_envs)]             # self._traj[e] is a List[Transition] - Re-initialize an empty list for every parallel environment
        self.states: List[object] = []                                                      # states is a List[object] - Reset the flattened 1D list that will hold all raw state objects
        self.actions: torch.Tensor | None = None                                            # If self.actions is not None then it is a torch.Tensor - Clear the finalized actions tensor
        self.rewards: torch.Tensor | None = None                                            # Clear the finalized rewards tensor
        self.dones: torch.Tensor | None = None                                              # Clear the finalized termination flags tensor
        self.log_probs: torch.Tensor | None = None                                          # Clear the finalized log-probabilities tensor
        self.values: torch.Tensor | None = None                                             # Clear the finalized state-values tensor
        self.advantages: torch.Tensor | None = None                                         # Clear the finalized GAE advantages tensor
        self.returns: torch.Tensor | None = None                                            # Clear the finalized discounted returns tensor

    def push(self, env_id: int, state, action: int, reward: float, done: float, log_prob: float, value: float) -> None:
        """
        Appends a single step transition to the designated environment's trajectory.

        Instantiates a `Transition` dataclass using the provided scalars and state object, 
        ensuring proper Python float/int casting. It then appends this object to the 
        specific sub-list inside `self._traj` matching the `env_id`.
        
        Args:
            env_id (int): The index of the parallel environment that generated this step.
            state (object): The observation representation (e.g., graph tensors).
            action (int): The chosen discrete action index.
            reward (float): The immediate scalar reward received.
            done (float): 1.0 if the episode ended, 0.0 otherwise.
            log_prob (float): The log-probability of the action under the acting policy.
            value (float): The critic's value estimate for the state.
            
        Returns:
            None.
            
        Example:
            >>> buffer = PPOBuffer(torch.device("cpu"), 1)
            >>> buffer.push(0, "state", 1, 0.5, 0.0, -1.2, 0.4)
            >>> len(buffer._traj[0])
            1
        """
        # ------------------------------------------------------------------------------------
        # Transition Logging
        # Cast transition variables to basic python types and append to the correct env track.
        # ------------------------------------------------------------------------------------
        self._traj[int(env_id)].append(Transition(state, int(action), float(reward), float(done), float(log_prob), float(value))) # Cast all inputs to precise types, construct a Transition, and append to the env's list

    def finalize(self, last_values: torch.Tensor) -> None:
        """
        Computes GAE, calculates returns, and flattens all collected trajectories into tensors.
        
        How it works:
        1. Formats the terminal value estimates used to bootstrap the final sequence step.
        2. Iterates over each environment's trajectory in reverse order.
        3. Computes the Temporal Difference (TD) error and accumulates it into the GAE score.
        4. Calculates the target returns (Advantage + Value).
        5. Flattens all parallel sequences into contiguous 1D PyTorch tensors and moves 
           them to the specified compute device for training.
        
        Args:
            last_values (torch.Tensor): A tensor of shape [num_envs] containing the Critic's 
            value estimates for the states immediately following the end of the rollouts.
        
        Returns:
            None. But updates the buffer's attributes: `self.states`, `self.actions`, `self.rewards`,
            `self.dones`, `self.log_probs`, `self.values`, `self.advantages`, and `self.returns` with finalized tensors.
            
        Example:
            >>> buffer = PPOBuffer(torch.device("cpu"), 1)
            >>> buffer.push(0, "state", 1, 1.0, 0.0, -0.5, 0.5)
            >>> last_vals = torch.tensor([0.2])
            >>> buffer.finalize(last_vals)
            >>> print(buffer.advantages.shape)
            torch.Size([1])
        """
        # ---------------------------------------------------------------------------------
        # Bootstrapping Preparation
        # Process the final value estimates needed to compute the last TD error.
        # ---------------------------------------------------------------------------------
        last_values = last_values.detach().float().to(self.device).view(-1)                 # Detach from graph, cast to float, move to target device, and flatten to a 1D tensor
        assert last_values.numel() == self.num_envs                                         # assert that last_values.shape == (self.num_envs,) - Ensure we have exactly one value per environment

        # ---------------------------------------------------------------------------------
        # Global List Initialization
        # Prepare empty python lists to accumulate data across all environments linearly.
        # ---------------------------------------------------------------------------------
        all_states: List[object] = []                                                       # Initialize an empty list to aggregate raw state objects from all parallel trajectories
        all_actions, all_rewards, all_dones, all_logps, all_vals = [], [], [], [], []       # Initialize multiple empty lists to collect scalars for actions, rewards, dones, logps, and values
        all_advs, all_rets = [], []                                                         # Initialize empty lists specifically for the computed advantages and returns

        # ------------------------------------------------------------------------------------
        # Per-Environment GAE Computation
        # Iterate over each environment's trajectory backwards, calculating TD errors and GAE.
        # ------------------------------------------------------------------------------------
        for e in range(self.num_envs):                                                      # Compute GAE per env - Loop explicitly through every parallel environment index
            traj = self._traj[e]                                                            # Extract the specific list of Transition objects logged for this environment
            if len(traj) == 0:                                                              # Check if the trajectory is completely empty (e.g., if env skipped collection)
                continue                                                                    # Bypass the calculation entirely for this empty trajectory sequence

            rewards = np.array([t.reward for t in traj], dtype=np.float32)                  # Extract all rewards into a contiguous NumPy float32 array for fast vector math
            dones = np.array([t.done for t in traj], dtype=np.float32)                      # Extract all done flags into a contiguous NumPy float32 array
            values = np.array([t.value for t in traj], dtype=np.float32)                    # Extract all critic value estimates into a contiguous NumPy float32 array

            adv = np.zeros_like(rewards)                                                    # Pre-allocate an empty NumPy array of identical shape to hold the computed advantages
            gae = 0.0                                                                       # Initialize the running Generalized Advantage Estimation accumulator to zero
            next_value = float(last_values[e].item())                                       # Extract the scalar bootstrap value for the step immediately following this sequence

            # Iterate backwards over the time dimension of the collected trajectory
            for i in reversed(range(len(rewards))):                                         
                # Compute continuation mask: 1.0 if episode continues, 0.0 if episode terminated here
                mask = 1.0 - dones[i]                                                       
                # TD error - Calculate the step-wise temporal difference error
                delta = rewards[i] + self.gamma * next_value * mask - values[i]             
                # GAE - Accumulate the exponentially discounted moving average of TD errors
                gae = delta + self.gamma * self.gae_lambda * mask * gae                     
                # Store the finalized running advantage value into the current timestep slot
                adv[i] = gae                                                                
                # Update next_value to the current step's value for the preceding step's calculation
                next_value = values[i]                                                      

            # Compute targeted return as advantage plus the baseline value
            ret = adv + values                                                              

            # -----------------------------------------------------------------------------
            # Cross-Environment Flattening
            # Append this environment's processed sequence to the global 1D tracking lists.
            # -----------------------------------------------------------------------------
            for i, t in enumerate(traj):                                                    # Loop sequentially forward through the processed trajectory and its computed metrics
                all_states.append(t.state)                                                  # Append the raw state object to the global aggregation list
                all_actions.append(t.action)                                                # Append the integer action to the global aggregation list
                all_rewards.append(t.reward)                                                # Append the float reward to the global aggregation list
                all_dones.append(t.done)                                                    # Append the float termination flag to the global aggregation list
                all_logps.append(t.log_prob)                                                # Append the initial log probability to the global aggregation list
                all_vals.append(t.value)                                                    # Append the baseline state value to the global aggregation list
                all_advs.append(float(adv[i]))                                              # Cast and append the computed NumPy advantage to the global Python float list
                all_rets.append(float(ret[i]))                                              # Cast and append the computed NumPy return to the global Python float list

        # ---------------------------------------------------------------------------------
        # PyTorch Tensor Finalization
        # Cast the fully flattened Python lists into PyTorch tensors on the active device.
        # ---------------------------------------------------------------------------------
        self.states = all_states                                                            # Bind the global list of raw state objects to the class attribute
        self.actions = torch.tensor(all_actions, dtype=torch.long, device=self.device)      # Convert aggregated actions list into a long integer tensor on the target device
        self.rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)   # Convert aggregated rewards list into a float32 tensor on the target device
        self.dones = torch.tensor(all_dones, dtype=torch.float32, device=self.device)       # Convert aggregated done flags list into a float32 tensor on the target device
        self.log_probs = torch.tensor(all_logps, dtype=torch.float32, device=self.device)   # Convert aggregated log probabilities list into a float32 tensor on the target device
        self.values = torch.tensor(all_vals, dtype=torch.float32, device=self.device)       # Convert aggregated baseline values list into a float32 tensor on the target device
        self.advantages = torch.tensor(all_advs, dtype=torch.float32, device=self.device)   # Convert aggregated advantages list into a float32 tensor on the target device
        self.returns = torch.tensor(all_rets, dtype=torch.float32, device=self.device)      # Convert aggregated returns list into a float32 tensor on the target device

    def get_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[torch.Tensor]:
        """
        Generates mini-batches of indices for iterating over the finalized buffer data.
        
        Calculates the total number of finalized transitions. If shuffling is requested, 
        it generates a randomized permutation of indices; otherwise, it creates a 
        sequential range. It then chunks this 1D tensor of indices into sub-tensors of size 
        `batch_size` and yields them consecutively using a Python generator.
        
        Args:
            batch_size (int): The requested size for each mini-batch (e.g., 64, 128).
            shuffle (bool, optional): Whether to randomly permute the data indices. Defaults to True.
            
        Returns:
            Iterator[torch.Tensor]: A Python generator yielding 1D tensors of integer indices.
            
        Example:
            >>> buffer = PPOBuffer(torch.device("cpu"), 1)
            >>> # ... (push data and finalize)
            >>> # Assume 10 items in buffer
            >>> for indices in buffer.get_batches(batch_size=5, shuffle=False):
            ...     print(indices)
            tensor([0, 1, 2, 3, 4])
            tensor([5, 6, 7, 8, 9])
        """
        # ---------------------------------------------------------------------------------
        # Batch Index Generation
        # Create randomized or sequential subsets of dataset indices for epoch training.
        # ---------------------------------------------------------------------------------
        assert self.actions is not None and self.advantages is not None and self.returns is not None and self.log_probs is not None # Verify that finalize() was called and tensors physically exist
        # Retrieve the total flat dataset size from the first dimension of the actions tensor
        n = self.actions.size(0)                                                            
        
        # Generate a tensor of indices: either shuffled or sequential based on the `shuffle` flag
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device) # shuffle indices if shuffle is True, else produce a linear sequential range tensor
        # Iterate sequentially over the flat dataset length in jumps of batch_size
        for start in range(0, n, batch_size):
            # Returns a slice of the index tensor from `start` to `start + batch_size`, yielding a mini-batch of indices
            yield idx[start:start + batch_size]                                             