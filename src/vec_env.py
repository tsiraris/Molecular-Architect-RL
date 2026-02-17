import torch
from chem_env import MoleculeEnvironment
from torch_geometric.data import Data
from rdkit import Chem

class VectorMoleculeEnv:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.envs = [MoleculeEnvironment(device) for _ in range(num_envs)]
        
    def reset(self):
        obs_list = []
        for env in self.envs:
            x, edge_index, edge_attr, _ = env.reset()
            obs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
        return obs_list

    def step(self, action_indices, curriculum_ratio=0.0):
        next_obs_list, rewards, dones, infos = [], [], [], [] # <--- Added infos list
        
        for i, env in enumerate(self.envs):
            # 1. Step the environment
            obs_tuple, reward, done = env.step(action_indices[i].item(), curriculum_ratio)
            
            info = {}
            if done:
                # CRITICAL FIX: Capture the molecule BEFORE resetting!
                try:
                    info['terminal_smiles'] = Chem.MolToSmiles(env.current_mol.GetMol())
                except:
                    info['terminal_smiles'] = "INVALID"
                
                # 2. NOW reset (after capturing the smiley)
                obs_tuple = env.reset()
            
            x, edge_index, edge_attr, _ = obs_tuple
            next_obs_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        # Return 4 values: obs, rewards, dones, INFOS
        return next_obs_list, \
               torch.tensor(rewards, dtype=torch.float32).to(self.device), \
               torch.tensor(dones, dtype=torch.float32).to(self.device), \
               infos 

    def get_masks(self):
        return torch.stack([env.get_action_mask() for env in self.envs])
    
    def get_smiles(self):
        return [Chem.MolToSmiles(env.current_mol.GetMol()) for env in self.envs]