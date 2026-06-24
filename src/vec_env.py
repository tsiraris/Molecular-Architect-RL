"""
Molecular Architect RL: Vectorized Environment Wrapper.

This script defines the `VectorMoleculeEnv` class, a crucial infrastructure component 
for reinforcement learning (specifically PPO). Deep RL algorithms require massive 
amounts of experience to learn stable policies. Executing a single environment sequentially 
is computationally inefficient, especially when using GPU-accelerated Neural Networks 
that thrive on batched data.

This class acts as an orchestrator, wrapping multiple independent `MoleculeEnvironment` 
instances into a single, unified interface. When the agent requests a step, this vector 
wrapper loops through the individual environments, applies each specific action, collects 
the resulting graph observations, rewards, and done flags, and batches them together.
Crucially, it implements an "auto-reset" mechanism: if an individual episode terminates, 
it automatically resets that specific environment and swaps in the new initial state, 
ensuring a continuous, uninterrupted flow of batched data for the learning algorithm.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch
from torch_geometric.data import Data
from chem_env import MoleculeEnvironment, ActionSpec


class VectorMoleculeEnv:
    """
    A vectorized environment manager for parallel molecular graph generation.
    
    Instantiates a list of standalone `MoleculeEnvironment` objects. It translates 
    batched tensor actions from the agent into individual step commands for each 
    environment. It then packages the individual outputs (node features, edge features) 
    into PyTorch Geometric `Data` objects, allowing them to be processed by a 
    `DataLoader` or `Batch` object downstream. It also handles automatic environment 
    resets to maintain fixed batch sizes during training.

    Args:
        num_envs (int): The number of parallel molecular environments to run.
        device (torch.device): The hardware device for tensor allocation.
        action_spec (ActionSpec | None, optional): Vocabulary constraints passed to envs.
        
    Example:
        >>> vec_env = VectorMoleculeEnv(num_envs=4, device=torch.device('cpu'))
        >>> obs_list = vec_env.reset()
        >>> len(obs_list)
        4
    """
    def __init__(self, num_envs: int, device: torch.device, action_spec: ActionSpec | None = None):
        # -----------------------------------------------------------------------------------------
        # Initialization and Instantiation
        # Store configuration and spawn the requested number of independent environments.
        # -----------------------------------------------------------------------------------------
        self.num_envs = num_envs                                                                    # Store the total number of parallel environments to maintain
        self.device = device                                                                        # Store the compute device for allocating observation tensors
        self.envs = [MoleculeEnvironment(device, action_spec=action_spec) for _ in range(num_envs)] # List[MoleculeEnvironment]
        self.num_actions = self.envs[0].num_actions                                                 # Extract the scalar action space size from the first environment to expose globally

    def reset(self) -> List[Data]:
        """
        Resets all parallel environments to their initial single-carbon states.
        
        Iterates through the internal list of environments, calls `reset()` on each, 
        and wraps the resulting PyTorch geometric feature tuples (x, edge_index, edge_attr) 
        into standardized `torch_geometric.data.Data` objects for easy batching.
        
        Args:
            None.
            
        Returns:
            List[Data]: A list of PyG Data objects representing the initial states of all envs.
            
        Example:
            >>> vec_env = VectorMoleculeEnv(4, torch.device('cpu'))
            >>> initial_states = vec_env.reset()
        """
        # -----------------------------------------------------------------------------------------
        # Global Reset Loop
        # Force all sub-environments back to zero and package their initial observations.
        # -----------------------------------------------------------------------------------------
        obs_list = []                                                                               # Initialize an empty Python list to accumulate the graph data objects
        for env in self.envs:                                                                       # Loop sequentially through every instantiated environment wrapper
            x, edge_index, edge_attr, _ = env.reset()                                               # Trigger a hard reset on the specific environment and unpack its returned tuple
            obs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))                  # Construct a PyG Data object from the raw tensors and append it to our batch list
        return obs_list                                                                             # List[Data] 

    def step(self, action_indices: torch.Tensor, curriculum_ratio: float = 0.0) -> Tuple[List[Data], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Advances all environments by one step using a batch of actions.
        
        Takes a tensor of discrete action indices corresponding to each environment. 
        It steps each environment individually. Crucially, if an environment reports 
        `done=True`, this method extracts its final SMILES string for logging, and 
        then automatically calls `reset()` on that specific environment. This auto-reset 
        replaces the terminal observation with a fresh initial state, allowing the 
        PPO loop to continue uninterrupted with a full batch.
        
        Args:
            action_indices (torch.Tensor): A 1D tensor of shape [num_envs] containing actions.
            curriculum_ratio (float, optional): The phase multiplier for the reward function.
            
        Returns:
            Tuple: Contains a list of next states (Data objects), a tensor of float rewards, 
            a tensor of float done flags, and a list of info dictionaries.
            
        Example:
            >>> vec_env = VectorMoleculeEnv(2, torch.device('cpu'))
            >>> vec_env.reset()
            >>> actions = torch.tensor([1, 4])
            >>> obs, rewards, dones, infos = vec_env.step(actions)
        """
        # -----------------------------------------------------------------------------------------
        # Step Iteration Setup
        # Initialize empty lists to collect outputs across the batch of environments.
        # -----------------------------------------------------------------------------------------
        next_obs_list: List[Data] = []                                                              # Prepare an empty list to store the resulting graph states after actions are taken
        rewards, dones = [], []                                                                     # Prepare empty lists to collect the scalar rewards and terminal flags
        infos: List[Dict[str, Any]] = []                                                            # Prepare an empty list to collect the metadata dictionaries

        # -----------------------------------------------------------------------------------------
        # Execution and Auto-Reset Loop
        # Loop through each environment and apply the corresponding action from the batch.
        # -----------------------------------------------------------------------------------------
        for i, env in enumerate(self.envs):                                                           # Iterate over environments alongside their index to map them to the correct action
            obs_tuple, reward, done, info = env.step(int(action_indices[i].item()), curriculum_ratio) # Extract the integer action for this env, execute the step, and catch the outputs

            # -------------------------------------------------------------------------------------
            # Auto-Reset Logic
            # If the episode has ended, reset the environment and capture any terminal information.
            # -------------------------------------------------------------------------------------
            if done:                                                                                # Evaluate if the underlying environment has reached a terminal condition (max steps or explicit stop)
                try:                                                                                # Wrap the SMILES extraction in a try block to prevent logging errors from halting training
                    info["terminal_smiles"] = env.get_smiles()                                      # Capture the final chemical structure as a string and inject it into the metadata info dict
                except Exception:                                                                   # Catch exceptions raised during unstable terminal topological translations
                    info["terminal_smiles"] = "INVALID"                                             # Inject a safe fallback string if RDKit cannot serialize the molecule
                obs_tuple = env.reset()                                                             # Immediately reset the terminated environment to get a fresh state, maintaining continuous batch flow

            # -------------------------------------------------------------------------------------
            # Batch Aggregation
            # Format raw tuple observations into Data objects and append metrics.
            # -------------------------------------------------------------------------------------
            x, edge_index, edge_attr, _ = obs_tuple                                                 # Unpack the observation tuple into its components
            next_obs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))             # Append the new observation as a Data object to the list
            rewards.append(float(reward))                                                           # Cast the reward to a float and append it to our batch tracker
            dones.append(float(done))                                                               # Cast the boolean done flag to a float (1.0 for True, 0.0 for False) for tensor compatibility
            infos.append(info)                                                                      # Append the augmented info dictionary to our batch tracker
            
        return (
            next_obs_list,
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
            infos,
        )                                                                                           # returns: (List[Data], torch.Tensor, torch.Tensor, List[Dict[str, Any]])

    def get_masks(self) -> torch.Tensor:
        """
        Aggregates the individual action masks from all sub-environments.
        
        Calls `get_action_mask()` on each underlying environment to retrieve a 1D 
        boolean tensor of legal moves. It then stacks these individual masks along 
        a new batch dimension (dim=0).
        
        Args:
            None.
            
        Returns:
            torch.Tensor: A 2D boolean tensor of shape [num_envs, num_actions] representing 
            the valid action mask for the current batch state.
            
        Example:
            >>> vec_env = VectorMoleculeEnv(4, torch.device('cpu'))
            >>> vec_env.reset()
            >>> batch_mask = vec_env.get_masks()
            >>> batch_mask.shape
            torch.Size([4, 119])
        """
        # ---------------------------------------------------------------------------------------------
        # Mask Batching: Query valid actions from all envs and stack them into a batched tensor block.
        # ---------------------------------------------------------------------------------------------
        return torch.stack([env.get_action_mask() for env in self.envs], dim=0)                     # Returns a tensor of action masks for all environments, where each mask indicates valid actions for the current state. 

    def get_smiles(self) -> List[str]:
        """
        Retrieves the SMILES representations for the current states of all environments.
        
        Executes a list comprehension querying `get_smiles()` on each environment instance. 
        Useful for logging or tracking the intermediate generation states.
        
        Args:
            None.
            
        Returns:
            List[str]: A list of SMILES strings, length equal to `num_envs`.
            
        Example:
            >>> vec_env = VectorMoleculeEnv(2, torch.device('cpu'))
            >>> vec_env.reset()
            >>> vec_env.get_smiles()
            ['C', 'C']
        """
        # -----------------------------------------------------------------------------------------
        # SMILES Batch Retrieval
        # -----------------------------------------------------------------------------------------
        return [env.get_smiles() for env in self.envs]                                              # Returns a list of SMILES strings representing the current molecules in all environments.