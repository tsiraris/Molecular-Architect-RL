import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch_geometric.data import Data
import numpy as np

class MoleculeEnvironment:
    def __init__(self, device):
        self.device = device
        self.current_mol = None
        self.focus_node_idx = 0 # TRACKER: Which atom are we currently looking at?
        
        # Actions: Add C, N, O, Stop, Shift Focus
        self.atom_types = ['C', 'N', 'O'] 
        self.num_actions = len(self.atom_types) + 2 # +1 for Stop, +1 for Shift Focus
    
    def reset(self):
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles('C')) 
        Chem.SanitizeMol(self.current_mol)
        self.focus_node_idx = 0
        return self._get_observation()

    def step(self, action_idx):
        # Action Indices:
        # 0..2: Add Atom (C, N, O)
        # 3: Stop
        # 4: Shift Focus (Jump to next valid atom)
        
        stop_idx = len(self.atom_types)
        shift_idx = len(self.atom_types) + 1

        # 1. STOP
        if action_idx == stop_idx:
            if self.current_mol.GetNumAtoms() < 3:
                return self._get_observation(), -5.0, True
            return self._get_observation(), self._calculate_reward(), True

        # 2. SHIFT FOCUS
        if action_idx == shift_idx:
            # Move focus to the next atom in the list (Cyclic)
            self.focus_node_idx = (self.focus_node_idx + 1) % self.current_mol.GetNumAtoms()
            # Small penalty to discourage infinite shifting without building
            return self._get_observation(), -0.1, False

        # 3. ADD ATOM
        atom_symbol = self.atom_types[action_idx]
        try:
            new_atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
            # Attach only to the Focus Node (not the last one)
            self.current_mol.AddBond(int(self.focus_node_idx), new_atom_idx, Chem.BondType.SINGLE)
            
            Chem.SanitizeMol(self.current_mol)
            
            # Moving focus to the new atom to encourage growth, agent can Shift back if it wants to branch.
            self.focus_node_idx = new_atom_idx
            
            reward = 0.0 
            done = False
        except Exception:
            reward = -1.0
            done = True 
        
        return self._get_observation(), reward, done
    
    def get_action_mask(self):
        mask = torch.ones(self.num_actions, dtype=torch.bool).to(self.device)
        
        # 1. Check STOP Constraint
        if self.current_mol.GetNumAtoms() < 3:
            mask[len(self.atom_types)] = 0 

        # 2. Check ADD Constraint (Based on FOCUS Node)
        focus_atom = self.current_mol.GetAtomWithIdx(int(self.focus_node_idx))
        try:
            current_valence = focus_atom.GetValence(Chem.ValenceType.EXPLICIT)
        except:
            current_valence = focus_atom.GetExplicitValence()
            
        symbol = focus_atom.GetSymbol()
        max_valence = 4 if symbol == 'C' else 3 if symbol == 'N' else 2
        
        if current_valence >= max_valence:
            # Cannot add to this node, it is full!
            for i in range(len(self.atom_types)):
                mask[i] = 0
        
        return mask

    def _calculate_reward(self):
        try:
            mol = self.current_mol.GetMol()
            return QED.qed(mol) * 10
        except:
            return 0.0

    def _get_observation(self):
        mol = self.current_mol
        atom_features = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            # --- PHYSICS INFORMED FEATURES (SOTA) ---
            
            # 1. Atom Type (One-Hot, 3 dims)
            symbol = atom.GetSymbol()
            type_feat = [1 if symbol == k else 0 for k in self.atom_types]
            
            # 2. Hybridization (One-Hot, 3 dims: SP, SP2, SP3)
            hyb = atom.GetHybridization()
            hyb_feat = [
                1 if hyb == Chem.rdchem.HybridizationType.SP else 0,
                1 if hyb == Chem.rdchem.HybridizationType.SP2 else 0,
                1 if hyb == Chem.rdchem.HybridizationType.SP3 else 0
            ]
            
            # 3. Aromaticity (Bool, 1 dim)
            arom_feat = [1] if atom.GetIsAromatic() else [0]
            
            # 4. Ring Membership (Bool, 1 dim)
            ring_feat = [1] if atom.IsInRing() else [0]
            
            # 5. FOCUS FLAG (Crucial: "Are you the active node?", 1 dim)
            focus_feat = [1] if i == self.focus_node_idx else [0]
            
            # Combine all: 3 + 3 + 1 + 1 + 1 = 9 Features
            atom_features.append(type_feat + hyb_feat + arom_feat + ring_feat + focus_feat)
            
        x = torch.tensor(atom_features, dtype=torch.float).to(self.device)

        rows, cols = [], []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            rows.extend([start, end])
            cols.extend([end, start])
            
        edge_index = torch.tensor([rows, cols], dtype=torch.long).to(self.device)
        
        # Handle edge case: single atom molecule has no bonds
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)

        batch = torch.zeros(len(atom_features), dtype=torch.long).to(self.device)
        return x, edge_index, batch