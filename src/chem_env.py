import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch_geometric.data import Data
import numpy as np

class MoleculeEnvironment:
    def __init__(self, device):
        self.device = device
        self.current_mol = None
        self.focus_node_idx = 0 # Focus tracker: Which atom we are currently looking at
        
        # Actions: Add C, N, O, Stop, Shift Focus
        self.atom_types = ['C', 'N', 'O'] 
        self.num_actions = len(self.atom_types) + 2 # +1 for Stop, +1 for Shift Focus
    
    def reset(self):
        """
        Resets the environment to the initial state (a single carbon atom) and returns the initial observation
        """
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles('C')) 
        Chem.SanitizeMol(self.current_mol)
        self.focus_node_idx = 0
        return self._get_observation()

    def step(self, action_idx):
        """
        Executes the given action and returns the new observation, reward, and done flag.
        """
        # Action Indices:
        # 0..2: Add Atom (C, N, O)
        # 3: Stop
        # 4: Shift Focus (Jump to next valid atom)
        
        stop_idx = len(self.atom_types)
        shift_idx = len(self.atom_types) + 1

        # 1. If STOP Action
        if action_idx == stop_idx:
            if self.current_mol.GetNumAtoms() < 3:
                return self._get_observation(), -5.0, True
            return self._get_observation(), self._calculate_reward(), True

        # 2. If SHIFT FOCUS Action
        if action_idx == shift_idx:
            # Move focus to the next atom in the list (Cyclic)
            self.focus_node_idx = (self.focus_node_idx + 1) % self.current_mol.GetNumAtoms()
            # Small penalty to discourage infinite shifting without building
            return self._get_observation(), -0.1, False

        # 3. If ADD ATOM Actions
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
        """
        Returns a boolean mask of valid actions based on the current molecule state and focus node.
        This is used to prevent the agent from taking invalid actions (e.g., adding to a full node or stopping too early).
        """       
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
        """
        Calculates the reward for the current molecule. For simplicity, QED is used as a proxy for drug-likeness, scaled by 10 to give a more substantial reward signal.        
        """
        try:
            mol = self.current_mol.GetMol()
            return QED.qed(mol) * 10
        except:
            return 0.0

    def _get_observation(self):
        """
        Converts the current molecule into a graph representation suitable for GNN input. 
        It creates node features based on atom type, hybridization, aromaticity, ring membership, and focus status. 
        Edge features are based on bond types: Edge attributes to inform the GNN about the bond type between atoms, and edge_index is constructed to represent the graph structure.
        The output is a tuple of (x, edge_index, edge_attr, batch) ready for PyTorch Geometric.
        """
        mol = self.current_mol
        atom_features = []
        
        # --- NODE FEATURES (Atoms) ---
        for i, atom in enumerate(mol.GetAtoms()):
            # 9 features: Type, Hyb, Arom, Ring, Focus
            symbol = atom.GetSymbol()
            type_feat = [1 if symbol == k else 0 for k in self.atom_types]
            hyb = atom.GetHybridization()
            hyb_feat = [
                1 if hyb == Chem.rdchem.HybridizationType.SP else 0,
                1 if hyb == Chem.rdchem.HybridizationType.SP2 else 0,
                1 if hyb == Chem.rdchem.HybridizationType.SP3 else 0
            ]
            arom_feat = [1] if atom.GetIsAromatic() else [0]        # Aromaticity
            ring_feat = [1] if atom.IsInRing() else [0]             # Ring membership
            focus_feat = [1] if i == self.focus_node_idx else [0]
            atom_features.append(type_feat + hyb_feat + arom_feat + ring_feat + focus_feat)  # Single list with 9 features in total per atom

        x = torch.tensor(atom_features, dtype=torch.float).to(self.device) # Shape: [num_atoms, 9]

        # --- EDGE FEATURES (Bonds) ---
        # Explicitly informing the GNN what kind of connection exists
        rows, cols = [], []
        edge_attr_list = []
        
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            rows.extend([start, end]) # "Source" atoms - Add both directions for undirected graph
            cols.extend([end, start]) # "Target" atoms
            
            # Encode Bond Type: Single, Double, Triple, Aromatic
            bt = bond.GetBondType()
            feat = [
                1 if bt == Chem.rdchem.BondType.SINGLE else 0,
                1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
                1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
                1 if bt == Chem.rdchem.BondType.AROMATIC else 0
            ] # [1,0,0,0] for single, [0,1,0,0] for double, etc.
            
            # Append feat for both directions (start->end and end->start)
            edge_attr_list.append(feat) 
            edge_attr_list.append(feat)

        # Topology of the graph: 2D matrix where top row is "Source" and bottom row is "Target".
        edge_index = torch.tensor([rows, cols], dtype=torch.long).to(self.device) # Graph structure (which atom is connected to which) - Shape: [2, num_edges] 2x the number of bonds due to undirected representation
        
        # Handle edge case: single atom molecule
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float).to(self.device) # 4 bond types
        else:
            # Chemistry of connections: a 4D feature vector for each edge to inform the GNN about the bond type between atoms
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(self.device) # Shape: [num_edges, 4 edge features (bond types)]

        batch = torch.zeros(len(atom_features), dtype=torch.long).to(self.device) # Single graph in the batch, so all zeros
        
        # RETURN TUPLE: x, edge_index, edge_attr, batch
        return x, edge_index, edge_attr, batch