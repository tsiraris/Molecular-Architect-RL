import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch_geometric.data import Data
import numpy as np

class MoleculeEnvironment:
    def __init__(self, device):
        self.device = device
        self.current_mol = None
        
        # Define the "Menu" of actions
        # 0: Add Carbon, 1: Add Nitrogen, 2: Add Oxygen, 3: Stop
        self.atom_types = ['C', 'N', 'O'] 
        self.num_actions = len(self.atom_types) + 1 
    
    def reset(self):
        """Start a new game with a single Carbon atom."""
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles('C')) # Creating a RWMol object from the RDKit library: a single Carbon atom for start, able to be modified
        Chem.SanitizeMol(self.current_mol)  # Ensure molecule is chemically valid
        return self._get_observation()

    def step(self, action_idx):
        """
        Apply the action chosen by the GNN.
        Returns: observation, reward, done (boolean)
        """
        # Scenario 1: Agent chooses STOP
        if action_idx == len(self.atom_types):
            # CONSTRAINT: The "Lazy Agent" Penalty - Punishment if the molecule is too small (e.g., just the starting Carbon), punish it.
            if self.current_mol.GetNumAtoms() < 3:
                reward = -5.0  # Big punishment for laziness
                done = True
                return self._get_observation(), reward, done
            
            # Otherwise, calculate real QED reward
            reward = self._calculate_reward()
            return self._get_observation(), reward, True

        # Scenario 2: The Agent chooses to ADD an ATOM
        # Simplified: Always add the new atom to the last atom added.
        atom_symbol = self.atom_types[action_idx]
        
        try:
            # RDKit magic to add an atom and a bond
            new_atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
            last_atom_idx = self.current_mol.GetNumAtoms() - 2
            self.current_mol.AddBond(last_atom_idx, new_atom_idx, Chem.BondType.SINGLE)
            
            # Check if the molecule is valid (Valency rules, e.g., Carbon needs 4 bonds)
            Chem.SanitizeMol(self.current_mol)
            
            # No intermediate reward for successfully adding an atom - Must finish the job.
            reward = 0.0 
            done = False
            
        except Exception:
            # If the move breaks chemistry rules (e.g., adding 5th bond to Carbon), punish it.
            reward = -1.0
            done = True # Game over
        
        return self._get_observation(), reward, done
    
    def get_action_mask(self):
        """
        Returns a boolean mask of valid actions.
        1 = Valid, 0 = Invalid.
        Shape: [num_actions]
        """
        # Always allow "Stop" (Last action)
        mask = torch.ones(self.num_actions, dtype=torch.bool).to(self.device)
        
        # CONSTRAINT: Disallow "Stop" if molecule is too small
        if self.current_mol.GetNumAtoms() < 3:  # "Stop" action is the last index
            mask[self.num_actions - 1] = 0  # Set Stop action to False
        
        # Get the "Focus" atom (The one we are attaching to) - The last added in simplification
        last_atom_idx = self.current_mol.GetNumAtoms() - 1
        last_atom = self.current_mol.GetAtomWithIdx(last_atom_idx)
        
        # Check explicit valence (How many bonds it already has)
        # New RDKit API to avoid warnings if possible, or stick to old
        try:
            current_valence = last_atom.GetValence(Chem.ValenceType.EXPLICIT)
        except:
            current_valence = last_atom.GetExplicitValence()
        
        # Get max allowed valence for this atom type (Simplified Rules)
        symbol = last_atom.GetSymbol()
        max_valence = 4 if symbol == 'C' else 3 if symbol == 'N' else 2
        
        # If the atom is "full", we CANNOT add more atoms to it.
        if current_valence >= max_valence:
            # Mask out all "Add Atom" actions (indices 0, 1, 2)
            # Only "Stop" (index 3) remains valid
            for i in range(len(self.atom_types)):
                mask[i] = 0
                
        return mask

    def _calculate_reward(self):
        """Final reward based on Drug Likeness (QED)."""
        try:
            mol = self.current_mol.GetMol()
            score = QED.qed(mol) # Returns 0.0 to 1.0
            return score * 10 # Scale it up to make it significant
        except:
            return 0.0

    def _get_observation(self):
        """
        Converts the RDKit molecule into PyTorch Geometric Data (x, edge_index)
        """
        mol = self.current_mol
        
        # 1. Get Node Features (One-Hot Encoding)
        # C=[1,0,0], N=[0,1,0], O=[0,0,1]
        atom_features = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'C': feat = [1, 0, 0]
            elif symbol == 'N': feat = [0, 1, 0]
            elif symbol == 'O': feat = [0, 0, 1]
            else: feat = [0, 0, 0] # Should not happen
            atom_features.append(feat)
            
        x = torch.tensor(atom_features, dtype=torch.float).to(self.device)

        # 2. Get Edge Index (Connectivity)
        rows, cols = [], []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            # Add both directions
            rows.extend([start, end])
            cols.extend([end, start])
            
        edge_index = torch.tensor([rows, cols], dtype=torch.long).to(self.device)
        
        # 3. Batch Vector (All atoms belong to molecule 0)
        batch = torch.zeros(len(atom_features), dtype=torch.long).to(self.device)

        return x, edge_index, batch