import torch
from rdkit import Chem
from rdkit.Chem import QED, GraphDescriptors, Descriptors
import numpy as np

class MoleculeEnvironment:
    """
    RL Environment for Molecule Generation.
    Uses Sigmoid Desirability Functions for SOTA Multi-Parameter Optimization.
    """
    def __init__(self, device, max_steps=30):
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0
        self.current_mol = None
        self.focus_node_idx = 0
        
        # Action Space: 11 Actions
        self.atom_types = ['C', 'N', 'O'] 
        self.bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
        self.num_actions = (len(self.atom_types) * len(self.bond_types)) + 2 
    
    def reset(self):
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles('C')) 
        Chem.SanitizeMol(self.current_mol)
        self.focus_node_idx = 0
        self.current_step = 0
        return self._get_observation()

    def step(self, action_idx, curriculum_ratio=0.0):
        self.current_step += 1
        num_atom_actions = len(self.atom_types) * len(self.bond_types)
        stop_idx = num_atom_actions
        shift_idx = num_atom_actions + 1

        if self.current_step >= self.max_steps:
            done = True
            reward = -5.0 if self.current_mol.GetNumAtoms() < 3 else self._calculate_reward(curriculum_ratio)
            return self._get_observation(), reward, done

        if action_idx == stop_idx:
            if self.current_mol.GetNumAtoms() < 3:
                return self._get_observation(), -5.0, True
            return self._get_observation(), self._calculate_reward(curriculum_ratio), True

        if action_idx == shift_idx:
            self.focus_node_idx = (self.focus_node_idx + 1) % self.current_mol.GetNumAtoms()
            return self._get_observation(), -0.1, False

        # Decode Action
        bond_type_idx = action_idx // len(self.atom_types) # Determine bond type index
        atom_type_idx = action_idx % len(self.atom_types)  # Determine atom type index
        
        try:
            new_atom_idx = self.current_mol.AddAtom(Chem.Atom(self.atom_types[atom_type_idx]))
            self.current_mol.AddBond(int(self.focus_node_idx), new_atom_idx, self.bond_types[bond_type_idx])
            Chem.SanitizeMol(self.current_mol)
            self.focus_node_idx = new_atom_idx
            reward = 0.0 
            done = False
        except:
            reward = -1.0 
            done = True 
        
        return self._get_observation(), reward, done
    
    def get_action_mask(self):
        mask = torch.ones(self.num_actions, dtype=torch.bool).to(self.device)
        if self.current_mol.GetNumAtoms() < 3:
            mask[len(self.atom_types) * len(self.bond_types)] = 0 

        focus_atom = self.current_mol.GetAtomWithIdx(int(self.focus_node_idx))
        try: val = focus_atom.GetExplicitValence()
        except: val = 0
            
        max_val = 4 if focus_atom.GetSymbol() == 'C' else (3 if focus_atom.GetSymbol() == 'N' else 2)
        remaining = max_val - val

        num_atom_actions = len(self.atom_types) * len(self.bond_types)
        for i in range(num_atom_actions):
            bond_idx = i // len(self.atom_types)
            required = bond_idx + 1 
            if required > remaining:
                mask[i] = 0
        return mask

    def _calculate_reward(self, curriculum_ratio):
        """Sigmoid-based MPO Reward."""
        try:
            mol = self.current_mol.GetMol()
            qed_score = QED.qed(mol)
            
            def sigmoid(value, target, scale=0.05):
                return 1.0 / (1.0 + np.exp(scale * (value - target)))

            mw_des = sigmoid(Descriptors.MolWt(mol), 500)
            logp_des = sigmoid(Descriptors.MolLogP(mol), 5)
            bertz_des = sigmoid(GraphDescriptors.BertzCT(mol), 800)
            
            # Geometric Mean
            mpo_score = (qed_score * mw_des * logp_des * bertz_des) ** (1/4)
            final_reward = mpo_score * 10.0
            
            # Curriculum
            if curriculum_ratio < 0.5:
                return (qed_score * 10.0) * (1 - curriculum_ratio) + (final_reward * curriculum_ratio)
            return final_reward
        except:
            return 0.0

    def _get_observation(self):
        mol = self.current_mol
        atom_features = []
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            type_feat = [1 if symbol == k else 0 for k in self.atom_types]
            hyb = atom.GetHybridization()
            hyb_feat = [1 if hyb == t else 0 for t in [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]]
            atom_features.append(type_feat + hyb_feat + [1 if atom.GetIsAromatic() else 0] + [1 if atom.IsInRing() else 0] + [1 if i == self.focus_node_idx else 0])

        x = torch.tensor(atom_features, dtype=torch.float).to(self.device)
        rows, cols, attr = [], [], []
        for bond in mol.GetBonds():
            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows.extend([s, e]); cols.extend([e, s])
            bt = bond.GetBondType()
            f = [1 if bt == t else 0 for t in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]]
            attr.extend([f, f])

        edge_index = torch.tensor([rows, cols], dtype=torch.long).to(self.device)
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float).to(self.device)
        else:
            edge_attr = torch.tensor(attr, dtype=torch.float).to(self.device)

        return x, edge_index, edge_attr, torch.zeros(len(atom_features), dtype=torch.long).to(self.device)