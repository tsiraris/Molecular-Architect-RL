from __future__ import annotations
from typing import Any, Dict, List, Tuple
import torch
from torch_geometric.data import Data
from chem_env import MoleculeEnvironment, ActionSpec


class VectorMoleculeEnv:
    """
    A vectorized environment that manages multiple instances of MoleculeEnvironment in parallel.
    This class provides a unified interface to reset and step through multiple environments simultaneously,
    and to retrieve action masks and SMILES strings for all environments.
    Args:
        num_envs (int): The number of parallel environments to manage.
        device (torch.device): The device on which tensors should be allocated.
        action_spec (ActionSpec | None): Optional specification for action space, passed to each environment.
    """
    def __init__(self, num_envs: int, device: torch.device, action_spec: ActionSpec | None = None):
        self.num_envs = num_envs
        self.device = device
        self.envs = [MoleculeEnvironment(device, action_spec=action_spec) for _ in range(num_envs)]  # List[MoleculeEnvironment]
        self.num_actions = self.envs[0].num_actions 

    def reset(self) -> List[Data]:
        """ Resets all environments and returns a list of initial observations as PyTorch Geometric Data objects. """
        obs_list = []
        for env in self.envs:
            x, edge_index, edge_attr, _ = env.reset()
            obs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return obs_list # List[Data] 

    def step(self, action_indices: torch.Tensor, curriculum_ratio: float = 0.0) -> Tuple[List[Data], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """ Takes a step in all environments using the provided action indices, and returns the next observations, rewards, done flags, and info dictionaries. """
        next_obs_list: List[Data] = []
        rewards, dones = [], []
        infos: List[Dict[str, Any]] = []

        # Loop through each environment and apply the corresponding action from the batch.
        for i, env in enumerate(self.envs):  
            obs_tuple, reward, done, info = env.step(int(action_indices[i].item()), curriculum_ratio)

            # If the episode has ended, reset the environment and capture any terminal information.
            if done:
                try:
                    info["terminal_smiles"] = env.get_smiles()
                except Exception:
                    info["terminal_smiles"] = "INVALID"
                obs_tuple = env.reset()

            x, edge_index, edge_attr, _ = obs_tuple     # Unpack the observation tuple into its components
            next_obs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr)) # Append the new observation as a Data object to the list
            rewards.append(float(reward))
            dones.append(float(done))
            infos.append(info)   
            
        return (
            next_obs_list,
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
            infos,
        ) # returns: (List[Data], torch.Tensor, torch.Tensor, List[Dict[str, Any]])

    def get_masks(self) -> torch.Tensor:
        """ Returns a tensor of action masks for all environments, where each mask indicates valid actions for the current state. """
        return torch.stack([env.get_action_mask() for env in self.envs], dim=0)

    def get_smiles(self) -> List[str]:
        """ Returns a list of SMILES strings representing the current molecules in all environments. """
        return [env.get_smiles() for env in self.envs]