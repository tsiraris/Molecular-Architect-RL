import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, QED, rdMolDescriptors

# ----------------------------
# Helper: simple structural alerts (very small starter set)
# ----------------------------
_ALERT_SMARTS = {
    "nitro": "[N+](=O)[O-]",
    "anilide": "NC(=O)c1ccccc1",
    "hydrazine": "NN",
    "azide": "N=[N+]=[N-]",
}
_ALERT_PATT = {k: Chem.MolFromSmarts(v) for k, v in _ALERT_SMARTS.items()}    # Pre-compile the SMARTS patterns into RDKit Mol objects for efficient substructure searching.


def _count_alerts(mol: Chem.Mol) -> int:
    """Count the number of structural alerts present in the molecule."""
    c = 0
    for patt in _ALERT_PATT.values():
        if patt is not None and mol.HasSubstructMatch(patt):
            c += 1
    return c


def _sigmoid_window(x: float, lo: float, hi: float, k: float = 6.0) -> float:
    """Smooth window ~1 inside [lo,hi], falls off outside."""
    # map to two sigmoids
    a = 1.0 / (1.0 + math.exp(k * (x - hi)))
    b = 1.0 / (1.0 + math.exp(k * (lo - x)))
    return a * b


@dataclass          # To store the vocabulary settings cleanly - Auto-generated __init__ and other methods, and can still have default values 
class ActionSpec:   
    """
    A translator between discrete action indices (input) and their semantic meaning in the molecule environment (output).
    Encapsulates the action space specification for the molecule environment, including the maximum number of atoms, allowed atom types, and bond types. 
    It also provides methods to decode action indices into meaningful actions and to calculate the total number of actions.
    """
    max_atoms: int = 25
    atom_types: Tuple[str, ...] = ("C", "N", "O", "F", "S", "Cl")  # Tightened action space: removed Br/I/P to prevent reward hacking & unrealistic chem
    bond_types: Tuple[Chem.BondType, ...] = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
    
    # Action space layout:
    # [0] = stop
    @property                 # To calculate the size of the vocabulary dynamically - Can access any method as an attribute (e.g., spec.n_stop) instead of a method call (e.g., spec.n_stop())
    def n_stop(self) -> int:  # Number of "stop" actions, which is always 1 (the action to terminate the episode).
        return 1

    # [1..n_add] = add actions (atom_type, bond_type) to the current focus
    @property
    def n_add(self) -> int:
        return len(self.atom_types) * len(self.bond_types) # returns the total number of "add" actions, each combination of atom type and bond type corresponds to a unique "add" action.

    # [n_add..n_add+n_focus] = focus actions (set focus to an existing atom index)
    @property
    def n_focus(self) -> int:
        return self.max_atoms  # returns the total number of "focus" actions available 

    # [n_add+n_focus..] = ring closure actions (close a ring between focus and an existing atom index with bond type)
    @property
    def n_ring(self) -> int:   
        return self.max_atoms * len(self.bond_types)

    # Total number of actions is the sum of all action types: stop, add, focus, and ring closure actions.
    @property
    def num_actions(self) -> int:
        return self.n_stop + self.n_add + self.n_focus + self.n_ring

    def decode(self, action_idx: int) -> Tuple[str, Dict]:
        """
        To translate the agent's numeric choices into chemical actions.
        Takes an action index and returns a tuple (action_type, payload):
        action_type in {"stop","add","focus","ring"} 
        payload is a dict with relevant info for that action type (e.g. atom_symbol and bond_type for "add", atom_index for "focus", etc.)
        """
        a = action_idx
        if a == 0:
            return "stop", {}
        a -= 1

        if a < self.n_add:
            bt_i = a // len(self.atom_types)  # Bond type index for the "add" action.
            at_i = a % len(self.atom_types)   # Atom type index 
            return "add", {"atom_symbol": self.atom_types[at_i], "bond_type": self.bond_types[bt_i]}
        a -= self.n_add  # Adjust the action index to account for the "add" actions, so that the remaining index (a) can be correctly interpreted as either a "focus" or "ring" action based on its value relative to n_focus and n_ring.

        if a < self.n_focus:
            return "focus", {"atom_index": a}
        a -= self.n_focus

        # Create a ring connection between the focus atom and an existing atom index with a specified bond type.
        bt_i = a // self.max_atoms  # Bond type index 
        j = a % self.max_atoms      # Target atom index (j) for the ring closure action.
        return "ring", {"atom_index": j, "bond_type": self.bond_types[bt_i]} # returns action type "ring", a payload that includes the target atom index (j) and the bond type for the ring closure, decoded from the given action index.


class MoleculeEnvironment:
    """
    Constrained molecular design environment.

    - State: RDKit molecule (RWMol) + focus atom index
    - Actions (fixed discrete head):
        0: stop
        1..: add atom (atom_type, bond_type) to focus
        ...: set focus to an existing atom index (0..max_atoms-1, masked if doesn't exist)
        ...: ring closure between focus and an existing atom index with bond type
    """
    def __init__(self, device: torch.device, max_steps: int = 40, action_spec: Optional[ActionSpec] = None, min_atoms: int = 5):
        self.device = device
        self.max_steps = max_steps
        self.min_atoms = min_atoms
        self.current_step = 0
        self.current_mol: Optional[Chem.RWMol] = None   # Current_mol can either be an RDKit RWMol object representing the current molecule or None if the environment has not been initialized.
        self.focus_node_idx: int = 0

        self.spec = action_spec or ActionSpec()         # If no ActionSpec is provided during initialization, a default ActionSpec instance will be created and assigned to self.spec.
        self.num_actions = self.spec.num_actions
        self.atom_types = list(self.spec.atom_types)
        self.bond_types = list(self.spec.bond_types)

    # -------- Core API --------
    def reset(self):
        """ Initialize a new episode with a single carbon atom. Returns initial observation."""
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles("C"))
        Chem.SanitizeMol(self.current_mol)
        self.focus_node_idx = 0
        self.current_step = 0
        return self._get_observation()

    def step(self, action_idx: int, curriculum_ratio: float = 0.0):
        """ 
        Apply the given action to the current molecule, update the focus, and calculate reward.
        Returns: (obs_tuple, reward, done, info=dict)
        """
        assert self.current_mol is not None, "Call reset() first."  # Ensure environment has been initialized. If not, raise an AssertionError with the message "Call reset() first."
        self.current_step += 1

        info: Dict = {"invalid_action": False, "terminated": False}

        # Episode length truncation
        if self.current_step >= self.max_steps:
            info["terminated"] = True
            return self._get_observation(), self._terminal_reward(curriculum_ratio), True, info

        action_type, payload = self.spec.decode(int(action_idx)) # Decode the given action index into corresponding action type and payload 
        # Stop action: terminate episode and calculate reward
        if action_type == "stop":
            info["terminated"] = True
            return self._get_observation(), self._terminal_reward(curriculum_ratio), True, info

        # Focus action: change the current focus to an existing atom index
        if action_type == "focus":
            j = int(payload["atom_index"])
            if j < self.current_mol.GetNumAtoms():
                self.focus_node_idx = j
                return self._get_observation(), -0.02, False, info
            # invalid focus
            info["invalid_action"] = True
            return self._get_observation(), -0.3, False, info

        # Add action: add a new atom and bond to the current focus
        if action_type == "add":
            try:
                atom_symbol = payload["atom_symbol"]
                bond_type = payload["bond_type"]
                if not self._can_add(bond_type):  # Check if the current focus atom can accept the new bond type (valence constraints).
                    info["invalid_action"] = True
                    return self._get_observation(), -0.5, False, info

                new_atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))
                self.current_mol.AddBond(int(self.focus_node_idx), int(new_atom_idx), bond_type)
                Chem.SanitizeMol(self.current_mol)
                self.focus_node_idx = int(new_atom_idx)
                # Tiny living reward encouraging growth, but not too much
                return self._get_observation(), 0.05, False, info
            except Exception:
                # invalid chemistry => terminate (hard fail)
                info["invalid_action"] = True
                info["terminated"] = True
                return self._get_observation(), -1.0, True, info
        
        # Ring closure action: create a bond between the focus atom and an existing atom index, forming a ring
        if action_type == "ring":
            try:
                j = int(payload["atom_index"])  # Target node
                bond_type = payload["bond_type"]
                if j >= self.current_mol.GetNumAtoms() or j == int(self.focus_node_idx):
                    info["invalid_action"] = True
                    return self._get_observation(), -0.3, False, info
                
                # Prevent trivial reward hacking: avoid tiny strained rings (3-4 member) which dominate reward hacks
                if self.current_mol.GetBondBetweenAtoms(int(self.focus_node_idx), j) is not None:
                    info["invalid_action"] = True
                    return self._get_observation(), -0.3, False, info
                
                # Ensure the new bond doesn't violate valence rules for the focus atom
                if not self._can_add(bond_type):
                    info["invalid_action"] = True
                    return self._get_observation(), -0.5, False, info

                # Also ensure target atom can accept this bond
                if not self._atom_can_accept(j, bond_type):
                    info["invalid_action"] = True
                    return self._get_observation(), -0.5, False, info

                self.current_mol.AddBond(int(self.focus_node_idx), j, bond_type)
                Chem.SanitizeMol(self.current_mol)
                # small positive reward for closing rings, but not too much to avoid reward hacking
                return self._get_observation(), 0.08, False, info
            except Exception:
                info["invalid_action"] = True
                return self._get_observation(), -0.5, False, info

        # If reach here, the action type is unrecognized (shouldn't happen if ActionSpec is consistent)
        info["invalid_action"] = True
        return self._get_observation(), -0.5, False, info

    def get_action_mask(self) -> torch.Tensor:
        """
        Returns a boolean mask of valid actions for the CURRENT state. Valid actions are True, invalid actions are False. 
        This is used to prevent the agent from taking actions that would lead to invalid chemistry or out-of-bounds errors, and to guide the learning process by only allowing meaningful actions in each state.        
        """
        assert self.current_mol is not None, "Call reset() first."
        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)
        n_atoms = self.current_mol.GetNumAtoms()

        # "Stop" action be valid only if the current molecule has at least min_atoms, encouraging the agent to grow a sufficiently large molecule before stopping. 
        mask[0] = n_atoms >= self.min_atoms 

        # Add actions masking
        hal = sum(1 for a in self.current_mol.GetAtoms() if a.GetSymbol() in ("F","Cl","Br","I"))   # Count the number of halogen atoms (F, Cl, Br, I) in the current molecule to enforce a constraint on halogen content, discouraging the agent from adding excessive halogens.
        p_count = sum(1 for a in self.current_mol.GetAtoms() if a.GetSymbol() == "P")               # Also count the number of phosphorus atoms to enforce a constraint on phosphorus content.
        s_count = sum(1 for a in self.current_mol.GetAtoms() if a.GetSymbol() == "S")               # Count the number of sulfur atoms to reduce sulfur stacking reward hacks.
        offset = 1
        # For each possible combination of bond type and atom type for the "add" action
        for bt_i, bt in enumerate(self.bond_types):         # bt = actual bond type, to check valence constraints for the "add" action | bt_i = bond type index in the bond_types list, to calculate the correct action index in the mask.
            for at_i, _sym in enumerate(self.atom_types):   # at_i = atom type index in the atom_types list | sym = actual atom symbol (e.g., "C", "N", etc.) needed to enforce constraints on halogen and phosphorus content for the "add" action.
                
                idx = offset + bt_i * len(self.atom_types) + at_i  # Calculates the corresponding action index in the overall action space and checks if that action is valid given the current state of the molecule.

                allow_sym = True
                
                if _sym in ("F","Cl","Br","I") and hal >= 1:   # Disallow another halogen addition if >=1 halogens in the molecule, to prevent halogen-heavy trivial reward hacking.
                    allow_sym = False
                
                if _sym == "P" and p_count > 1:    # Disallow (excessive) phosphorus addition, to prevent phosphorus-heavy reward hacking.
                    allow_sym = False

                if _sym == "S" and s_count >= 2:   # Disallow sulfur stacking above 2
                    allow_sym = False
                
                mask[idx] = allow_sym and self._can_add(bt)     # "Add" action becomes valid if not violating valence constraints for the focus atom and halogen/phosphorus content constraints (checked by allow_sym).
        offset += self.spec.n_add                               # Update the offset to account for the "add" actions, so that the subsequent indices in the mask can be correctly interpreted as "focus" and "ring" actions based on their position relative to n_focus and n_ring.

        # Focus actions masking
        for j in range(self.spec.max_atoms):
            mask[offset + j] = j < n_atoms   # "Focus" action to set focus to an existing atom index j is valid only if j < current number of atoms in the molecule.
        offset += self.spec.n_focus          # Update the offset

        # Ring actions masking
        for bt_i, bt in enumerate(self.bond_types):  
            for j in range(self.spec.max_atoms):
                idx = offset + bt_i * self.spec.max_atoms + j
                if j < n_atoms and j != int(self.focus_node_idx):  # Only consider ring closure actions that connect the focus atom to a different existing atom index j, avoiding invalid self-connections and ensure the target atom exists in the molecule.
                    # Ring sanity: avoid tiny strained rings (3-4 member) which dominate reward hacks
                    try:
                        # Get shortest path between focus and target atom (j). If a bond is added between these, it would create a ring. 
                        # The length of this path indicates the size of the ring that would be formed by adding this bond.
                        path = Chem.rdmolops.GetShortestPath(self.current_mol, int(self.focus_node_idx), int(j))  
                        # New ring size would be len(path) + 1 (closing an edge)
                        if len(path) + 1 < 5:
                            continue
                    except Exception:   # If there's an error in getting the shortest path (e.g., if the molecule is in an invalid state), 
                        pass            # ignore ring sanity check for this action, as long as it doesn't violate valence constraints and isn't a self-connection. 
                    if self.current_mol.GetBondBetweenAtoms(int(self.focus_node_idx), j) is None:
                        # NOTE: Ring closure does NOT add an atom, so atom-type constraints (halogen/P limits) do not apply here.
                        # Instead, check valence for BOTH atoms (focus and target) to ensure the bond is feasible.
                        if self._atom_can_accept(int(self.focus_node_idx), bt) and self._atom_can_accept(int(j), bt):
                            mask[idx] = True
        return mask 

    # -------- Reward --------
    def _terminal_reward(self, curriculum_ratio: float) -> float:
        """ Calculate reward at episode termination. 
        Curriculum ratio (0.0-1.0) controls weighting between simple QED reward and full MPO reward, to ease early learning.
        """
        # Hard minimum size constraint
        if self.current_mol is None or self.current_mol.GetNumAtoms() < self.min_atoms:
            return -5.0

        r = self._calculate_reward()
        # Curriculum: start with QED-heavy, then shift to full MPO
        if curriculum_ratio <= 0.0:
            return 10.0 * self._safe_qed()
        if curriculum_ratio >= 1.0:
            return r
        return (1.0 - curriculum_ratio) * (10.0 * self._safe_qed()) + curriculum_ratio * r

    def _safe_qed(self) -> float:
        """ 
        Calculate QED reward, but catch any errors and return 0 if the molecule is in an invalid state.
        Preventing the entire reward calculation from crashing.
        """
        try:
            return float(QED.qed(self.current_mol.GetMol()))
        except Exception:
            return 0.0

    def _calculate_reward(self) -> float:
        """
        Multi-parameter objective:
        - Drug-likeness (QED)
        - Property windows (MW, cLogP, TPSA, HBD/HBA)
        - Complexity control (Bertz)
        - Structural alerts penalty
        - Ring sanity (avoid extreme ring counts)
        """
        try:
            mol = self.current_mol.GetMol()
            Chem.SanitizeMol(mol)

            qed = float(QED.qed(mol))                           # Quantitative Estimate of Drug-likeness (QED) 
            mw = float(Descriptors.MolWt(mol))                  # Molecular weight (MW) of the molecule.
            logp = float(Crippen.MolLogP(mol))                  # Octanol-water partition coefficient (cLogP), a measure of the molecule's hydrophobicity.
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))        # Topological Polar Surface Area (TPSA), which is a descriptor related to the molecule's ability to interact with polar solvents and is often used as a proxy for properties like solubility and permeability.
            hbd = int(Lipinski.NumHDonors(mol))                 # Number of hydrogen bond donors (HBD) in the molecule (= number of atoms that can donate a hydrogen bond, e.g. -OH, -NH groups).
            hba = int(Lipinski.NumHAcceptors(mol))              # Number of hydrogen bond acceptors (HBA) in the molecule (= number of atoms that can accept a hydrogen bond, e.g. O, N atoms).
            rings = int(rdMolDescriptors.CalcNumRings(mol))     # Total number of rings in the molecule (indicator of molecular complexity and influences properties like rigidity and synthetic accessibility).
            bertz = float(GraphDescriptors.BertzCT(mol))        # Bertz complexity, a topological descriptor that quantifies the complexity of the molecule's structure based on its graph representation ---> higher values indicating more complex molecules.

            # Counts / extra descriptors for realism
            hal = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))     # Count the number of halogen atoms (F, Cl, Br, I) in the molecule.
            p_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "P")                    # Count the number of phosphorus atoms in the molecule.
            rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))                              # Number of rotatable bonds, which is a measure of the molecule's flexibility and can influence properties like binding affinity and bioavailability.
            heavy = int(mol.GetNumHeavyAtoms())                                                 # Number of heavy atoms (non-hydrogen atoms) in the molecule, a simple measure of molecular size and complexity.
            arom_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))                        # Number of aromatic rings in the molecule, for properties like stability, reactivity, and interactions with biological targets.
            sp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))                                 # Fraction of sp3-hybridized carbons with respect to total carbon count, a measure of the molecule's three-dimensionality and can influence properties like solubility and metabolic stability.
            bridge = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))                          # Number of bridgehead atoms (=part of two or more rings), can indicate the presence of fused or bridged ring systems, contributing to molecular complexity.
            spiro = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))                                # Number of spiro atoms (shared by two rings), can indicate the presence of spirocyclic structures, which are often found in bioactive molecules and can contribute to unique three-dimensional shapes.
            s_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "S")                    # Sulfur count used to reduce sulfur-stacking loopholes.

            # Smooth cut-off for desirability windows (rough medchem targets)
            d_mw = _sigmoid_window(mw, 180.0, 520.0)
            d_logp = _sigmoid_window(logp, 0.5, 4.5)
            d_tpsa = _sigmoid_window(tpsa, 20.0, 120.0)
            d_hbd = _sigmoid_window(hbd, 0.0, 5.0)
            d_hba = _sigmoid_window(hba, 0.0, 10.0)
            d_rings = _sigmoid_window(rings, 1.0, 6.0)
            d_bertz = 1.0 / (1.0 + math.exp(0.004 * (bertz - 900.0)))


            # Extra realism terms
            # Halogens: allow a couple, penalize halogen-heavy reward hacking
            d_hal = _sigmoid_window(float(hal), 0.0, 1.0)   # tightened to <=1 halogen to avoid halogen collapse in practice
            # "SA-like" proxy: prefers moderate size, moderate rotatable bonds, moderate aromaticity, not-too-flat
            d_rot = _sigmoid_window(float(rot), 0.0, 8.0)
            d_heavy = _sigmoid_window(float(heavy), 15.0, 45.0)
            d_arom = _sigmoid_window(float(arom_rings), 0.0, 4.0)
            d_sp3 = _sigmoid_window(float(sp3), 0.15, 0.85)
            d_bridge = _sigmoid_window(float(bridge), 0.0, 2.0)
            d_spiro = _sigmoid_window(float(spiro), 0.0, 2.0)

            # Sulfur penalty term to reduce repeated sulfur stacking reward hacks (soft penalty; mask also enforces a hard cap)
            d_sulfur = _sigmoid_window(float(s_count), 0.0, 2.0)
            
            # Single synthetic accessibility (SA) proxy score (moderate size, complexity), using a geometric mean.
            # If any factor is very unfavorable (close to 0), it will significantly reduce the overall score, while still allowing for some compensation if other factors are favorable. 
            d_sa_proxy = float(np.exp(np.mean(np.log(np.clip([d_rot, d_heavy, d_arom, d_sp3, d_bridge, d_spiro], 1e-6, 1.0))))) # np.clip  to ensure that the individual components are between 1e-6 and 1.0 (not zero).
            # Hard discouragement of phosphorus-heavy chemistry (common loophole)
            p_pen = math.exp(-1.25 * float(p_count))  # kept for safety if seeding ever introduces P, even though P is removed from action space

            alerts = _count_alerts(mol)
            alert_pen = math.exp(-1.0 * alerts)  # 1.0 if none, downweights if alerts exist

            # Combine (geometric mean-ish)
            components = [max(qed, 1e-6), d_mw, d_logp, d_tpsa, d_hbd, d_hba, d_rings, d_bertz, alert_pen, d_hal, d_sa_proxy, p_pen, d_sulfur]
            # Overall multi-parameter objective (MPO) score is calculated as the geometric mean of the individual component scores, which include drug-likeness (QED), property desirability windows (MW, cLogP, TPSA, HBD/HBA, ring count, Bertz complexity), structural alerts penalty, and additional realism factors (halogen content, synthetic accessibility proxy, phosphorus penalty).
            mpo = float(np.exp(np.mean(np.log(np.clip(components, 1e-6, 1.0))))) 

            # Scale
            return 12.0 * mpo
        except Exception:
            return 0.0

    # -------- Chemistry constraints --------
    def _atom_max_valence(self, atom: Chem.Atom) -> int:
        """ Return a conservative estimate of the maximum valence for the given atom. """
        sym = atom.GetSymbol()
        # Conservative typical valences (not exhaustive)
        if sym == "C":
            return 4
        if sym == "N":
            return 3
        if sym == "O":
            return 2
        if sym in ("F", "Cl", "Br", "I"):
            return 1
        if sym == "S":
            return 6  # allows hypervalence, still constrained by sanitize
        if sym == "P":
            return 5
        return 4

    def _atom_can_accept(self, atom_idx: int, bond_type: Chem.BondType) -> bool:
        """ Check if the atom at atom_idx can accept a new bond of the given bond_type without violating valence constraints."""
        atom = self.current_mol.GetAtomWithIdx(int(atom_idx))
        max_v = self._atom_max_valence(atom)
        try:
            v = atom.GetExplicitValence()  # returns the number of bonds the atom currently has
        except Exception:
            v = 0
        required = 1 if bond_type == Chem.BondType.SINGLE else (2 if bond_type == Chem.BondType.DOUBLE else 3)
        return (max_v - v) >= required  # Returns True if the available valence is >= to the required valence for the new bond (atom can accept the new bond). Otherwise, it returns False (violates valence rules for that atom).

    def _can_add(self, bond_type: Chem.BondType) -> bool:
        """ Check if we can add a new atom with the given bond type to the current focus without violating valence constraints or size cap."""
        # size cap
        if self.current_mol.GetNumAtoms() >= self.spec.max_atoms:
            return False
        return self._atom_can_accept(int(self.focus_node_idx), bond_type) # True only if the focus atom can accept the new bond type (valence constraints) and the total number of atoms in the molecule is less than the maximum allowed (max_atoms). 

    def _get_observation(self):
        """
        Convert the current molecule into a graph representation suitable for GNN input.
        Node features: one-hot atom type, hybridization, aromaticity, ring membership, focus
        Edge features: one-hot bond type
        Returns (x, edge_index, edge_attr, node_mask) where:
        - x: [num_atoms, num_node_features] tensor of node features
        - edge_index: [2, num_edges] tensor of edge indices (source and target node indices for each edge)
        - edge_attr: [num_edges, num_edge_features] tensor of edge features
        - node_mask: [num_atoms] boolean tensor indicating which nodes are valid (useful for padding to max_atoms) - in this implementation, all nodes are valid since we don't pad.
        """
        assert self.current_mol is not None
        mol = self.current_mol

        atom_features: List[List[float]] = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            type_feat = [1.0 if symbol == k else 0.0 for k in self.atom_types]   # One-hot encoding of the atom type based on the predefined list of allowed atom types in the ActionSpec.

            hyb = atom.GetHybridization()
            hyb_feat = [1.0 if hyb == t else 0.0 for t in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)]

            atom_features.append(type_feat + hyb_feat + [1.0 if atom.GetIsAromatic() else 0.0, 1.0 if atom.IsInRing() else 0.0, 1.0 if i == int(self.focus_node_idx) else 0.0,])

        x = torch.tensor(atom_features, dtype=torch.float32, device=self.device) # shape [num_atoms, num_node_features=len(type_feat)+len(hyb_feat)+3]

        rows, cols, attr = [], [], []
        for bond in mol.GetBonds():
            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows.extend([s, e])  # Add both directions for undirected graph representation (source and target node indices for each edge).
            cols.extend([e, s])
            bt = bond.GetBondType()
            f = [1.0 if bt == t else 0.0 for t in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)]
            attr.extend([f, f])  # undirected edges, so add both directions with the same features

        if len(rows) == 0: # Handle case with no bonds (e.g., single atom) to avoid empty tensors
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        else: 
            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=self.device)   # shape [2, num_edges], where the first row contains source node indices and the second row contains target node indices for each edge in the molecule. 
            edge_attr = torch.tensor(attr, dtype=torch.float32, device=self.device)         # shape [num_edges, num_edge_features=4], where each row corresponds to the one-hot encoding of the bond type for that edge (single, double, triple, aromatic).

        # keep API-compatible 4th return
        return x, edge_index, edge_attr, torch.zeros(x.size(0), dtype=torch.long, device=self.device)   # node_mask (not needed here since we don't pad, but can be used in future if we implement padding to max_atoms)

    # -------- Convenience --------
    def get_smiles(self) -> str:
        """ 
        Get current molecule as SMILES string, for logging/debugging. 
        Returns "INVALID" if the molecule is in an invalid state that can't be converted to SMILES. 
        """
        try:
            return Chem.MolToSmiles(self.current_mol.GetMol())
        except Exception:
            return "INVALID"