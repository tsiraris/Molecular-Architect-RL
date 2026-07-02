"""
=========================================
Constrained Molecular Design Environment
========================================

This script defines the `MoleculeEnvironment` and `ActionSpec` classes, which together 
form the environment of a graph reinforcement-learning pipeline for de novo small-molecule generation.
It provides a rigorous Phase-1 generative baseline where an actor-critic agent builds a 
2D molecular graph atom-by-atom. The agent modifies the molecule by selecting 
from a discrete set of actions: stopping, adding an atom and bond, shifting the focus node, 
or closing a ring.

Crucially, the environment enforces chemical grammar via valency-based action masking. 
Before the agent samples an action, the environment computes a boolean mask of chemically 
legal moves, ensuring the policy never proposes physically impossible chemistry that would 
waste compute or crash RDKit. The environment also provides a robust 
reward system using a multi-parameter objective (MPO) that guides the agent toward 
drug-like topologies by penalizing structural alerts and constraining descriptors like 
QED, MW, logP, TPSA, and Bertz complexity.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, QED, rdMolDescriptors
from reward.synth_gate import synth_gate, BANNED_PENALTY   # Stage-1 synthesizability gate
from reward.composite import combine as _combine_reward, default_reward_cfg as _default_reward_cfg  # Stage-2 affinity+diversity combiner

# -----------------------------------------------------------------------------------------
# Global Structural Alerts
# These dictionaries pre-compile SMARTS patterns for specific toxic or undesirable 
# structural functional groups. This allows fast pattern matching during reward calculation.
# -----------------------------------------------------------------------------------------
_ALERT_SMARTS = {                                                                           # Initialize a dictionary mapping alert names to their SMARTS string representations
    "nitro": "[N+](=O)[O-]",                                                                # Define the SMARTS string for a nitro group
    "anilide": "NC(=O)c1ccccc1",                                                            # Define the SMARTS string for an anilide group
    "hydrazine": "NN",                                                                      # Define the SMARTS string for a hydrazine group
    "azide": "N=[N+]=[N-]",                                                                 # Define the SMARTS string for an azide group
}                                                                                           # Close the SMARTS dictionary definition
_ALERT_PATT = {k: Chem.MolFromSmarts(v) for k, v in _ALERT_SMARTS.items()}                  # Pre-compile the SMARTS patterns into RDKit Mol objects for efficient substructure searching.


def _count_alerts(mol: Chem.Mol) -> int:
    """
    Counts the number of undesirable structural alerts present in the given molecule.
    
    Iterates through a pre-compiled dictionary of SMARTS patterns (`_ALERT_PATT`) and 
    uses RDKit's `HasSubstructMatch` to check if the molecule contains any of these 
    toxic or unstable functional groups. It tallies all matches found.
    
    Args:
        mol (Chem.Mol): The RDKit molecule object to be evaluated.
        
    Returns:
        int: The total count of matched structural alerts.
        
    Example:
        >>> m = Chem.MolFromSmiles("c1ccccc1N=[N+]=[N-]") # Phenyl azide
        >>> _count_alerts(m)
        1
    """
    c = 0                                                                                   # Initialize the alert counter to zero
    for patt in _ALERT_PATT.values():                                                       # Loop over each compiled RDKit SMARTS pattern in the dictionary values
        if patt is not None and mol.HasSubstructMatch(patt):                                # Check if the pattern is valid and if the molecule contains this substructure
            c += 1                                                                          # Increment the alert counter if a match is found
    return c                                                                                # Return the final count of structural alerts found


def _sigmoid_window(x: float, lo: float, hi: float, k: float = 6.0) -> float:
    """
    Calculates a smoothed window function that outputs ~1 inside [lo, hi] and falls off outside.
    
    Maps the input to the product of two opposing sigmoid functions. One sigmoid decays 
    as x exceeds the upper bound (`hi`), and the other decays as x falls below the 
    lower bound (`lo`). The parameter `k` controls the steepness of the drop-off.
    
    Args:
        x (float): The input property value to evaluate.
        lo (float): The lower bound of the desired window.
        hi (float): The upper bound of the desired window.
        k (float, optional): The steepness of the sigmoid curve. Defaults to 6.0.
        
    Returns:
        float: A continuous score between 0.0 and 1.0.
        
    Example:
        >>> score = _sigmoid_window(250.0, 180.0, 520.0) # Within bounds
        >>> round(score, 2)
        1.0
    """
    a = 1.0 / (1.0 + math.exp(k * (x - hi)))                                                # Compute the upper-bound sigmoid drop-off factor
    b = 1.0 / (1.0 + math.exp(k * (lo - x)))                                                # Compute the lower-bound sigmoid drop-off factor
    return a * b                                                                            # Multiply both components to get a window strictly near 1 inside [lo, hi]


@dataclass                                                                                  # To store the vocabulary settings cleanly - Auto-generated __init__ and other methods, and can still have default values 
class ActionSpec:   
    """
    A translator between discrete action indices (input) and their semantic meaning in the molecule environment (output).
    
    Encapsulates the action space specification for the molecule environment, including the maximum number of atoms, 
    allowed atom types, and bond types. It maps a flat 1D integer index to one of four blocks:
    Stop, Add (atom+bond), Focus (shift pointer), and Ring (close loop).
    
    Args:
        max_atoms (int): The maximum number of atoms allowed in the generated molecule. Defaults to 25.
        atom_types (Tuple[str, ...]): Allowed elemental symbols. Defaults to ("C", "N", "O", "F", "S", "Cl").
        bond_types (Tuple[Chem.BondType, ...]): Allowed bond types. Defaults to (SINGLE, DOUBLE, TRIPLE).
        
    Example:
        >>> spec = ActionSpec(max_atoms=10)
        >>> print(spec.num_actions)
        59
    """
    max_atoms: int = 25                                                                                         # Define the maximum allowed number of atoms in the generated molecule
    atom_types: Tuple[str, ...] = ("C", "N", "O", "F", "S", "Cl")                                               # Tightened action space: removed Br/I/P to prevent reward hacking & unrealistic chem
    bond_types: Tuple[Chem.BondType, ...] = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)  # Define the allowed bond types for additions and ring closures
    
    @property                                                                               # To calculate the size of the vocabulary dynamically - Can access any method as an attribute (e.g., spec.n_stop) instead of a method call (e.g., spec.n_stop())
    def n_stop(self) -> int:
        """
        Retrieves the number of available 'stop' actions.
        
        Returns a hardcoded value of 1, as there is exactly one action designated 
        to immediately terminate the episode.
        
        Args:
            None.
            
        Returns:
            int: The size of the stop action block (1).
            
        Example:
            >>> spec = ActionSpec()
            >>> spec.n_stop
            1
        """
        return 1                                                                            # Number of "stop" actions, which is always 1 (the action to terminate the episode).

    @property                                                                               # Define as a property so it behaves like an attribute
    def n_add(self) -> int:
        """
        Calculates the total number of 'add' actions in the vocabulary.
        
        Multiplies the number of possible atom types by the number of possible bond types,
        representing all possible permutations of adding a new atom with a specific bond.
        
        Args:
            None.
            
        Returns:
            int: The size of the add action block.
            
        Example:
            >>> spec = ActionSpec(atom_types=("C", "N"), bond_types=(Chem.BondType.SINGLE,))
            >>> spec.n_add
            2
        """
        return len(self.atom_types) * len(self.bond_types)                                  # returns the total number of "add" actions, each combination of atom type and bond type corresponds to a unique "add" action.

    @property                                                                               # Define as a property so it behaves like an attribute
    def n_focus(self) -> int:
        """
        Retrieves the total number of 'focus' actions in the vocabulary.
        
        Maps directly to the `max_atoms` limit, as the agent must be able to select 
        any valid atom index in the molecule as the new focal attachment point.
        
        Args:
            None.
            
        Returns:
            int: The size of the focus action block.
            
        Example:
            >>> spec = ActionSpec(max_atoms=15)
            >>> spec.n_focus
            15
        """
        return self.max_atoms                                                               # returns the total number of "focus" actions available 

    @property                                                                               # Define as a property so it behaves like an attribute
    def n_ring(self) -> int:
        """
        Calculates the total number of 'ring closure' actions in the vocabulary.
        
        Multiplies the maximum possible target atom indices (`max_atoms`) by the number of 
        available bond types, accounting for all possible bonds back to existing nodes.
        
        Args:
            None.
            
        Returns:
            int: The size of the ring closure action block.
            
        Example:
            >>> spec = ActionSpec(max_atoms=10, bond_types=(Chem.BondType.SINGLE,))
            >>> spec.n_ring
            10
        """
        return self.max_atoms * len(self.bond_types)                                        # Multiply max nodes by bond types to get total possible ring combinations

    @property                                                                               # Define as a property so it behaves like an attribute
    def num_actions(self) -> int:
        """
        Calculates the total size of the flat action space.
        
        Sums the sizes of the stop, add, focus, and ring closure action blocks.
        
        Args:
            None.
            
        Returns:
            int: The total count of all possible discrete actions.
            
        Example:
            >>> spec = ActionSpec()
            >>> type(spec.num_actions)
            <class 'int'>
        """
        # Total Action Vocabulary Size
        return self.n_stop + self.n_add + self.n_focus + self.n_ring                        # Sum all action block sizes to get the total size of the action space

    def decode(self, action_idx: int) -> Tuple[str, Dict]:
        """
        Translates a flat action index into a semantic action command.
        
        Sequentially subtracts the size of each action block from the index. Whichever block 
        causes the index to underflow determines the action type. Modulo and integer division 
        are then used to unpack the exact atom/bond/index payload from the remainder.
        
        Args:
            action_idx (int): The discrete integer action selected by the policy.
            
        Returns:
            Tuple[str, Dict]: A tuple containing the string action type ("stop", "add", 
            "focus", or "ring") and a dictionary payload of relevant parameters.
            
        Example:
            >>> spec = ActionSpec()
            >>> action_type, payload = spec.decode(0)
            >>> print(action_type)
            stop
        """
        # ---------------------------------------------------------------------------------
        # Action Index Decoding Sequence
        # Subtract block sizes sequentially to locate the correct action category.
        # ---------------------------------------------------------------------------------
        a = action_idx                                                                      # Copy the input action index into a local variable 'a' for sequential decoding
        if a == 0:                                                                          # Check if the action index corresponds to the 'stop' action (index 0)
            return "stop", {}                                                               # Return the "stop" action type and an empty payload dictionary
        a -= 1                                                                              # Decrement 'a' by 1 to offset the 'stop' action for subsequent checks

        if a < self.n_add:                                                                  # Check if the remaining index 'a' falls within the range of 'add' actions
            bt_i = a // len(self.atom_types)                                                # Bond type index for the "add" action.
            at_i = a % len(self.atom_types)                                                 # Atom type index 
            return "add", {"atom_symbol": self.atom_types[at_i], "bond_type": self.bond_types[bt_i]} # Return the "add" action with the resolved atom symbol and bond type payload
        a -= self.n_add                                                                     # Adjust the action index to account for the "add" actions, so that the remaining index (a) can be correctly interpreted as either a "focus" or "ring" action based on its value relative to n_focus and n_ring.

        if a < self.n_focus:                                                                # Check if the remaining index 'a' falls within the range of 'focus' actions
            return "focus", {"atom_index": a}                                               # Return the "focus" action type with the target atom index as the payload
        a -= self.n_focus                                                                   # Subtract the number of focus actions to evaluate the remaining index as a ring action

        # ---------------------------------------------------------------------------------
        # Ring Action Decoding
        # Extract bond type and target atom index for a ring closure.
        # ---------------------------------------------------------------------------------
        bt_i = a // self.max_atoms                                                          # Bond type index 
        j = a % self.max_atoms                                                              # Target atom index (j) for the ring closure action.
        return "ring", {"atom_index": j, "bond_type": self.bond_types[bt_i]}                # returns action type "ring", a payload that includes the target atom index (j) and the bond type for the ring closure, decoded from the given action index.


class MoleculeEnvironment:
    """
    Constrained molecular design environment utilizing an RDKit backend.
    
    Maintains a 2D graph state representing the current molecule and a focal attachment 
    index. It ingests discrete actions to modify the graph, enforces valency and geometry 
    rules via dynamic masking, and calculates shaped multi-objective rewards to guide PPO
    training towards drug-like structures.
    
    Args:
        device (torch.device): Compute device for outputting tensor observations/masks.
        max_steps (int, optional): Maximum sequence length per episode. Defaults to 40.
        action_spec (Optional[ActionSpec], optional): Vocabulary constraints. Defaults to None.
        min_atoms (int, optional): Minimum size before early stopping is permitted. Defaults to 5.
        
    Example:
        >>> env = MoleculeEnvironment(torch.device("cpu"))
        >>> obs = env.reset()
        >>> len(obs)
        4
    """
    def __init__(self, device: torch.device, max_steps: int = 40, action_spec: Optional[ActionSpec] = None, min_atoms: int = 5,
                 affinity_scorer=None, diversity_archive=None, reward_cfg: Optional[dict] = None):
        """
        Initializes the molecular environment configuration.
        
        Binds compute devices, episode bounds, and the action specification logic. 
        Sets up internal state variables (like the current molecule and step counter) to None/zero.
        
        Args:
            device (torch.device): Compute device for tensor operations.
            max_steps (int): The episode truncation limit.
            action_spec (Optional[ActionSpec]): Custom vocabulary specification object.
            min_atoms (int): The minimum allowable size for a valid terminal molecule.
            
        Returns:
            None
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cuda"))
            >>> env.max_steps
            40
        """
        self.device = device                                                                # Store the torch device for tensor allocations
        self.max_steps = max_steps                                                          # Store the maximum allowed steps per episode
        self.min_atoms = min_atoms                                                          # Store the minimum threshold of atoms required before stopping is rewarded
        # --- Stage-2 (optional): target-aware affinity + diversity reward components ---
        # Loads a pre-trained affinity scorer (surrogate/predict.py) and diversity archive (reward/composite.py) objects, if provided
        self.affinity_scorer = affinity_scorer                                              # AffinityScorer or None; queried at terminal when reward_cfg["use_affinity"] is True
        self.diversity_archive = diversity_archive                                          # DiversityArchive or None; supplies the anti-collapse Tanimoto penalty
        # reward_cfg provides the default hyperparameter configuration for the Stage-2 reward system (reward/composite.py), 
        # if not provided in the constructor the default values are explicitly tuned to exactly reproduce the Stage-1 behavior (affinity and diversity turned off).
        self.reward_cfg = reward_cfg if reward_cfg is not None else _default_reward_cfg()   # Stage-2 weights; defaults reproduce Stage-1 exactly
        self.last_reward_info = {}                                                          # Diagnostics from the most recent terminal reward (for logging)
        self.current_step = 0                                                               # Initialize the internal episode step counter to zero
        self.current_mol: Optional[Chem.RWMol] = None                                       # Current_mol can either be an RDKit RWMol object representing the current molecule or None if the environment has not been initialized.
        self.focus_node_idx: int = 0                                                        # Initialize the focal attachment pointer to atom index 0

        self.spec = action_spec or ActionSpec()                                             # If no ActionSpec is provided during initialization, a default ActionSpec instance will be created and assigned to self.spec.
        self.num_actions = self.spec.num_actions                                            # Extract and store the total scalar size of the action vocabulary
        self.atom_types = list(self.spec.atom_types)                                        # Convert the allowed atom tuple into a list for easier indexing
        self.bond_types = list(self.spec.bond_types)                                        # Convert the allowed bond tuple into a list for easier indexing

    # -------- Core API --------
    def reset(self):
        """
        Resets the environment to a fresh state containing a single Carbon atom.
        
        Instantiates a new RDKit RWMol from the SMILES "C", sanitizes it, resets the 
        focus pointer and step counters, and computes the initial graph observation.
        
        Args:
            None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The initial graph 
            observation block containing node features, edge indices, edge features, and mask.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> obs = env.reset()
        """
        self.current_mol = Chem.RWMol(Chem.MolFromSmiles("C"))                              # Initialize a new editable RDKit molecule from a basic Carbon SMILES
        Chem.SanitizeMol(self.current_mol)                                                  # Explicitly sanitize the fresh molecule to compute implicit valences
        self.focus_node_idx = 0                                                             # Reset the focal pointer to the starting carbon atom (index 0)
        self.current_step = 0                                                               # Reset the internal episode step tracker back to zero
        return self._get_observation()                                                      # Process and return the PyTorch Geometric observation tensors for the initial state

    def step(self, action_idx: int, curriculum_ratio: float = 0.0):
        """
        Executes a single step in the environment by applying the chosen action.
        
        Decodes the integer action into a chemical command, applies it to the RDKit `RWMol` 
        if legally valid, updates internal pointers, computes intermediate or terminal rewards, 
        and flags termination states based on action limits or the explicit 'stop' action.
        
        Args:
            action_idx (int): The flat index representing the agent's chosen action.
            curriculum_ratio (float, optional): Progression scalar (0.0 to 1.0) shaping the 
            complexity of the terminal reward function. Defaults to 0.0.
            
        Returns:
            Tuple: Contains the new observation tuple, a float reward, a boolean done flag, 
            and a dictionary of step metadata.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> obs, rew, done, info = env.step(1) # Add an atom
        """
        # ---------------------------------------------------------------------------------
        # Step Validation & Pre-processing
        # Assert initialization, increment steps, and handle maximum length truncation.
        # ---------------------------------------------------------------------------------
        assert self.current_mol is not None, "Call reset() first."                          # Ensure environment has been initialized. If not, raise an AssertionError with the message "Call reset() first."
        self.current_step += 1                                                              # Increment the internal step tracking counter by 1 for this turn

        info: Dict = {"invalid_action": False, "terminated": False}                         # Initialize the info dict to track structural validity and episode termination

        if self.current_step >= self.max_steps:                                             # Check if the episode length has hit the configured maximum limit
            info["terminated"] = True                                                       # Mark the episode as naturally terminated within the info dictionary
            return self._get_observation(), self._terminal_reward(curriculum_ratio), True, info # Return the final state, calculated terminal reward, a True done flag, and the info

        action_type, payload = self.spec.decode(int(action_idx))                            # Decode the given action index into corresponding action type and payload 
        
        # ---------------------------------------------------------------------------------
        # Action Branch - Stop
        # Terminate the episode immediately upon request.
        # ---------------------------------------------------------------------------------
        if action_type == "stop":                                                                   # Check if the decoded action is an explicit halt command
            info["terminated"] = True                                                               # Flag the info dictionary that the episode has terminated properly
            return self._get_observation(), self._terminal_reward(curriculum_ratio), True, info     # Return current observation, final evaluated reward, done=True, and info

        # ---------------------------------------------------------------------------------
        # Action Branch - Focus Shift
        # Move the attachment pointer to branch the molecule instead of making chains.
        # ---------------------------------------------------------------------------------
        if action_type == "focus":                                                          # Check if the decoded action is a focal pointer shift
            j = int(payload["atom_index"])                                                  # Extract the target integer atom index from the decoded payload
            if j < self.current_mol.GetNumAtoms():                                          # Ensure the targeted focus index actually exists within the current molecule bounds
                self.focus_node_idx = j                                                     # Update the internal focus pointer to the newly requested index
                return self._get_observation(), -0.02, False, info                          # Return new observation, a tiny penalty to discourage infinite shifting, and done=False
            info["invalid_action"] = True                                                   # If out of bounds, flag the info dict that the action was invalid
            return self._get_observation(), -0.3, False, info                               # Return the unchanged observation, a moderate penalty, and done=False

        # ---------------------------------------------------------------------------------
        # Action Branch - Add Atom
        # Attach a new atom via a designated bond to the currently focused atom.
        # ---------------------------------------------------------------------------------
        if action_type == "add":                                                            # Check if the action intends to append a new atom to the graph
            try:                                                                            # Wrap the addition logic in a try block to catch deep RDKit chemical exceptions
                atom_symbol = payload["atom_symbol"]                                        # Extract the elemental symbol string from the payload
                bond_type = payload["bond_type"]                                            # Extract the desired RDKit bond type object from the payload
                if not self._can_add(bond_type):                                            # Check if the current focus atom can accept the new bond type (valence constraints).
                    info["invalid_action"] = True                                           # Flag the action as structurally invalid if valence capacity is insufficient
                    return self._get_observation(), -0.5, False, info                       # Return unchanged observation, a hefty penalty, and done=False

                new_atom_idx = self.current_mol.AddAtom(Chem.Atom(atom_symbol))             # Instantiate the new RDKit atom and insert it into the RWMol, returning its index
                self.current_mol.AddBond(int(self.focus_node_idx), int(new_atom_idx), bond_type) # Link the newly created atom to the focal atom using the specified bond type
                Chem.SanitizeMol(self.current_mol)                                          # Re-sanitize the updated graph to recalculate valences and aromatics
                self.focus_node_idx = int(new_atom_idx)                                     # Automatically shift the focus pointer to the newly added atom
                return self._get_observation(), 0.05, False, info                           # Tiny living reward encouraging growth, but not too much
            except Exception:                                                               # Catch any underlying unrecoverable chemical graph errors thrown by RDKit
                info["invalid_action"] = True                                               # Flag that the action was definitively invalid
                info["terminated"] = True                                                   # Abort the episode entirely due to the hard chemical failure
                return self._get_observation(), -1.0, True, info                            # Return unchanged observation, max penalty, done=True, and info
        
        # ---------------------------------------------------------------------------------
        # Action Branch - Ring Closure
        # Form a cyclical bond between the focal atom and a historical graph node.
        # ---------------------------------------------------------------------------------
        if action_type == "ring":                                                           # Check if the decoded action is a ring cyclization
            try:                                                                            # Wrap ring closure in a try block to intercept complex graph routing failures
                j = int(payload["atom_index"])                                              # Target node
                bond_type = payload["bond_type"]                                            # Extract the requested bond type to close the ring with
                
                # Ensure the target node exists and is not the focal node itself (no self-loops)
                if j >= self.current_mol.GetNumAtoms() or j == int(self.focus_node_idx):    
                    info["invalid_action"] = True                                           # Flag the action as invalid due to out-of-bounds or self-referential topology
                    return self._get_observation(), -0.3, False, info                       # Return the unchanged observation, a moderate penalty, and done=False
                
                # Ensure a bond between the focal and target atom doesn't already exist
                if self.current_mol.GetBondBetweenAtoms(int(self.focus_node_idx), j) is not None: 
                    info["invalid_action"] = True                                           # Flag the action as invalid because a bond already spans this specific atom pair
                    return self._get_observation(), -0.3, False, info                       # Return unchanged observation, a penalty, and done=False
                
                # Ensure the new bond doesn't violate valence rules for the focus atom (can support the required bond order)
                if not self._can_add(bond_type):                                            
                    info["invalid_action"] = True                                           # Flag invalidity if the focal atom cannot support the required bond order
                    return self._get_observation(), -0.5, False, info                       # Return unchanged observation, heavy penalty, and done=False

                # Also ensure target atom can accept this bond
                if not self._atom_can_accept(j, bond_type):                                         
                    info["invalid_action"] = True                                           # Flag invalidity if the target ring atom cannot support the requested bond order
                    return self._get_observation(), -0.5, False, info                       # Return unchanged observation, heavy penalty, and done=False

                self.current_mol.AddBond(int(self.focus_node_idx), j, bond_type)            # Connect the focal atom and the target atom with the specified bond type
                Chem.SanitizeMol(self.current_mol)                                          # Recalculate RDKit topological properties, ring counts, and valences
                return self._get_observation(), 0.08, False, info                           # Small positive reward for closing rings, but not too much to avoid reward hacking
            except Exception:                                                               # Catch any severe failure during the sanitization of the new cyclic system
                info["invalid_action"] = True                                               # Flag the action as causing a critical chemical failure
                return self._get_observation(), -0.5, False, info                           # Return unchanged observation, a heavy penalty, and continue (done=False)

        # ---------------------------------------------------------------------------------
        # Unrecognized Action Fallback
        # ---------------------------------------------------------------------------------
        info["invalid_action"] = True                                                       # If reach here, the action type is unrecognized (shouldn't happen if ActionSpec is consistent)
        return self._get_observation(), -0.5, False, info                                   # Return unchanged observation, heavy penalty, and done=False

    def get_action_mask(self) -> torch.Tensor:
        """
        Computes a boolean mask indicating which actions are legally permissible.
        
        Enforces chemical grammar. It evaluates every action in the vocabulary 
        against the current graph state. It checks size limits, element-specific limits 
        (halogens <= 1, sulfur <= 2), remaining valence capacity of specific nodes, 
        and topological sanity checks (e.g., rejecting highly strained 3-to-4-membered rings).
        
        Args:
            None.
            
        Returns:
            torch.Tensor: A 1D boolean tensor of size `num_actions`, where True signifies 
            a valid, legal move.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> mask = env.get_action_mask()
            >>> mask.shape
            torch.Size([119])
        """
        # ---------------------------------------------------------------------------------
        # Action Mask Initialization
        # Prepare the zeroed mask tensor and evaluate the terminal stop condition.
        # ---------------------------------------------------------------------------------
        assert self.current_mol is not None, "Call reset() first."                          # Halt execution if a graph has not yet been initialized via reset()
        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)          # Allocate a full-zero boolean tensor on the compute device matching the action space size
        n_atoms = self.current_mol.GetNumAtoms()                                            # Query the current total atom count of the molecule
        # Stop action is only valid if the current molecule has at least min_atoms
        mask[0] = n_atoms >= self.min_atoms                                                 # Encouraging the agent to grow a sufficiently large molecule before stopping. 

        # ---------------------------------------------------------------------------------
        # Add Actions Masking
        # Mask additions based on elemental composition caps and valence capacities.
        # ---------------------------------------------------------------------------------
        
        # Count how many Halogens (F, Cl, Br, I), Phosphorus (P), and Sulfur (S) atoms are currently in the molecule.
        hal = sum(1 for a in self.current_mol.GetAtoms() if a.GetSymbol() in ("F","Cl","Br","I")) # Count the number of halogen atoms (F, Cl, Br, I) in the current molecule to enforce a constraint on halogen content, discouraging the agent from adding excessive halogens.
        p_count = sum(1 for a in self.current_mol.GetAtoms() if a.GetSymbol() == "P")       # Also count the number of phosphorus atoms to enforce a constraint on phosphorus content.
        s_count = sum(1 for a in self.current_mol.GetAtoms() if a.GetSymbol() == "S")       # Count the number of sulfur atoms to reduce sulfur stacking reward hacks.
        offset = 1                                                                          # Initialize offset counter to bypass the stop action (index 0)
        # Iterate over all possible bond types and atom types of the current molecule to evaluate the validity of 
        # each "add" action in the vocabulary (to fill the mask with which add actions are valid and which are not). 
        for bt_i, bt in enumerate(self.bond_types):                                         # bt = actual bond type, to check valence constraints for the "add" action | bt_i = bond type index in the bond_types list, to calculate the correct action index in the mask.
            for at_i, _sym in enumerate(self.atom_types):                                   # at_i = atom type index in the atom_types list | sym = actual atom symbol (e.g., "C", "N", etc.) needed to enforce constraints on halogen and phosphorus content for the "add" action.
                
                # Calculate the corresponding action index in the overall action space for this specific combination of bond type and atom type. 
                # Add actions are arranged in blocks where each block corresponds to a specific bond type, and within each block, the actions are ordered by atom type.
                idx = offset + bt_i * len(self.atom_types) + at_i                           # Calculates the corresponding action index in the overall action space and checks if that action is valid given the current state of the molecule.

                allow_sym = True                                                            # Assume the symbol is allowed by default until specific caps invalidate it
                
                # Disallow another halogen addition if >=1 halogens in the molecule, to prevent halogen-heavy trivial reward hacking.
                if _sym in ("F","Cl","Br","I") and hal >= 1:                                
                    allow_sym = False                                                       # Flip allowance flag to false if halogen limits are breached
                
                # Disallow (excessive) phosphorus addition, to prevent phosphorus-heavy reward hacking.
                if _sym == "P" and p_count > 1:                                             
                    allow_sym = False                                                       # Flip allowance flag to false if phosphorus limits are breached

                # Disallow sulfur stacking above 2, to prevent sulfur-heavy reward hacking.
                if _sym == "S" and s_count >= 2:                                            
                    allow_sym = False                                                       # Flip allowance flag to false if sulfur stacking limits are breached
                
                # If the focal atom can accept the proposed bond type (valence constraints), 
                # and the symbol is allowed, mark this action as valid in the mask.
                mask[idx] = allow_sym and self._can_add(bt)                                 # "Add" action becomes valid if not violating valence constraints for the focus atom and halogen/phosphorus content constraints (checked by allow_sym).
        # Update the offset to account for the "add" actions
        offset += self.spec.n_add                                                           # So that the subsequent indices in the mask can be correctly interpreted as "focus" and "ring" actions based on their position relative to n_focus and n_ring.

        # ---------------------------------------------------------------------------------
        # Focus Actions Masking
        # Mask out-of-bounds focal shifts.
        # ---------------------------------------------------------------------------------
        for j in range(self.spec.max_atoms):                                                # Loop through all theoretically possible absolute indices up to max capacity
            # Focus actions are only valid if the target atom index j is less than the current number of atoms in the molecule.
            mask[offset + j] = j < n_atoms                                                  
        offset += self.spec.n_focus                                                         # Update the offset

        # ---------------------------------------------------------------------------------
        # Ring Actions Masking
        # Mask impossible loops based on geometry (shortest path) and endpoint valency.
        # ---------------------------------------------------------------------------------
        
        # Iterate over all bond types and all possible target atom indices to evaluate the validity of each "ring closure" action in the vocabulary.
        for bt_i, bt in enumerate(self.bond_types):                                         # Loop through every allowable RDKit bond type
            for j in range(self.spec.max_atoms):                                            # Loop through every absolute node index serving as the target
                # Resolve the exact linear index mapping for this specific ring closure configuration
                idx = offset + bt_i * self.spec.max_atoms + j                               
                # If target atom exists and is not the focal atom itself, perform additional checks for ring closure validity.
                if j < n_atoms and j != int(self.focus_node_idx):                           
                    try:                                                                    # Wrap ring sanity checks in a try block to handle graph analysis failures
                        # Get shortest path between focus and target atom (j). If a bond is added between these, it would create a ring. 
                        # The length of this path plus 1 indicates the size of the ring that would be formed by adding this bond.
                        path = Chem.rdmolops.GetShortestPath(self.current_mol, int(self.focus_node_idx), int(j))  
                        if len(path) < 5:                                                   # New ring size equals len(path): path atoms form the ring once the closing edge is added
                        # Allowable ring sizes are 5 or greater; 3-4 membered rings mark this action as invalid in the mask.
                            continue                                                        # Skip marking as True if the proposed ring is too small (e.g., 3-4 membered)
                    except Exception:                                                       # If there's an error in getting the shortest path (e.g., if the molecule is in an invalid state), 
                        pass                                                                # Ignore ring sanity check for this action, as long as it doesn't violate valence constraints and isn't a self-connection. 
                    
                    # If the target atom is not already connected to the focal atom
                    if self.current_mol.GetBondBetweenAtoms(int(self.focus_node_idx), j) is None: 
                        # Note: Ring closure does NOT add an atom, so atom-type constraints (halogen/P limits) do not apply here.
                        # Instead, check valence for BOTH atoms (focus and target) to ensure the bond is feasible.
                        if self._atom_can_accept(int(self.focus_node_idx), bt) and self._atom_can_accept(int(j), bt): # Validate that BOTH the source and target atoms have sufficient free valence
                            mask[idx] = True                                                # Mark the specific ring closure action as totally valid
        return mask                                                                         # Return the fully computed boolean mask tensor

    # -------- Reward --------
    def _terminal_reward(self, curriculum_ratio: float) -> float:
        """
        Calculates the final reward score upon episode termination with a strict synthesizability gate.
        
        Enforces a hard minimum-size penalty. Extracts the RDKit molecule and runs it through a 
        strict synthesizability gate (`synth_gate`) to check for banned structural exploits 
        (like unstable S#C motifs). If the molecule contains illegal motifs or has a terrible 
        SA score, it returns an immediate heavy penalty (`BANNED_PENALTY`), overriding the 
        standard reward to prevent reward hacking. Otherwise, it blends the simple QED objective 
        and the full MPO objective via a linear curriculum, and scales the result by a soft 
        synthesizability multiplier.
        
        Args:
            curriculum_ratio (float): A scalar from 0.0 (early training) to 1.0 (late training).
            
        Returns:
            float: The calculated terminal reward scalar, gated by synthetic realism.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> r = env._terminal_reward(0.5)
        """
        # Check hard bounds to penalize the agent if it stops building too early.
        if self.current_mol is None or self.current_mol.GetNumAtoms() < self.min_atoms:     # Hard minimum size constraint check to ensure valid graph topology exists
            return -5.0                                                                     # Return an aggressive flat penalty if the molecule ended prematurely

        # Pull the pure topological graph to evaluate synthetic realism.
        mol = self.current_mol.GetMol()                                                     # Extract the standard RDKit Mol object from the read-write instance

        # Synthesizability Gate: Prevent 2D descriptor hacking by checking for fundamentally unstable chemistry (such as banned motifs, or SA>6, or PAINS).
        hard_ok, soft, _ginfo = synth_gate(mol)                                             # Pass the molecule through the external rule-based synthesizer filter
        # Soft-scale Realism Application: Apply soft as the final fractional modifier to penalize difficult but valid synthesis.
        # Reminder: soft is a Linearly decay reward for moderate complexity, exponentially decay for toxic alerts from PAINS and BRENK catalogues. It is a float in [0, 1] where 1 means fully valid and 0 means banned.
        if not hard_ok:                                                                     # Evaluate the hard pass/fail boolean flag returned by the gate
            return BANNED_PENALTY                                                           # Immediately yield bounded penalty for banned-motif or SA>6 junk chemistry

        # Curriculum Blending: Mix simple heuristic rewards (QED) with hard descriptors (MPO) over training time.
        r = self._calculate_reward()                                                        # Delegate to the complex multi-parameter objective evaluator to get full MPO score
        if curriculum_ratio <= 0.0:                                                         # Check if the training is in the earliest purely exploratory stage
            base = 10.0 * self._safe_qed()                                                  # Establish base reward strictly as 10x the robust QED calculation
        elif curriculum_ratio >= 1.0:                                                       # Check if the training curriculum has fully matured
            base = r                                                                        # Establish base reward strictly as the finalized full MPO calculation
        else:                                                                               # Handle the intermediate transitional phase of the curriculum
            # Linearly interpolate between the pure QED score and the full MPO objective (r)
            base = (1.0 - curriculum_ratio) * (10.0 * self._safe_qed()) + curriculum_ratio * r 
        # Stage-2 composite: fold in an optional affinity term (with uncertainty penalty) and a diversity
        # penalty. With use_affinity/use_diversity False (the defaults), this returns exactly base * soft.
        aff_hat_z = aff_unc_z = None
        # If the affinity surrogate is available ("use_affinity" = True), score the molecule 
        # (surrogate/predict.py: aff_hat_z = mean affinity and aff_unc_z = std z-scored pChEMBL from the deep ensemble)
        if self.reward_cfg.get("use_affinity") and self.affinity_scorer is not None:         # Only score when affinity is enabled and a surrogate is attached
            aff_hat_z, aff_unc_z = self.affinity_scorer.score(mol)                           # (mean, std) predicted z-scored pChEMBL from the deep ensemble
        diversity_pen = 0.0
        # If the diversity archive is available ("use_diversity" = True), returns a penalty for being similar to recently generated molecules 
        # Reminder: reward/composite.py/def penalty calculates the average Tanimoto similarity between the molecule and
        # the entire current rolling archive (fingerprints of all the generated molecules in this queue). 
        if self.reward_cfg.get("use_diversity") and self.diversity_archive is not None:      # Anti mode-collapse: penalise similarity to recently generated molecules
            diversity_pen = self.diversity_archive.penalty(mol)                              # Mean Tanimoto to the rolling archive (0 novel .. 1 duplicate)
            # Add the current molecule's fingerprint to the archive
            self.diversity_archive.add(mol)                                                  # Register this molecule so future ones are compared against it
        # Final Stage-2 composite reward and info dict (P, A, D, aff_hat_z, aff_unc_z)
        reward, info = _combine_reward(base, soft, curriculum_ratio, self.reward_cfg,        # Weighted gate x property x affinity(-beta*unc) x diversity combiner
                                       aff_hat_z=aff_hat_z, aff_unc_z=aff_unc_z, diversity_pen=diversity_pen)
        self.last_reward_info = info                                                         # Stash diagnostics (P, A, D, aff_hat_z, aff_unc_z) for the trainer to log
        return reward                                                                        # Return the Stage-2 composite terminal reward (== base*soft when affinity/diversity off)

    def _safe_qed(self) -> float:
        """
        Safely computes the Quantitative Estimate of Drug-likeness (QED).
        
        Attempts to compute the RDKit QED score. Wraps the execution in a try-except block 
        to ensure unstable topological states do not crash the environment.
        
        Args:
            None.
            
        Returns:
            float: The QED score [0, 1] or 0.0 if computation fails.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> qed_val = env._safe_qed()
        """
        try:                                                                                # Wrap evaluation to intercept internal RDKit analytical failures
            return float(QED.qed(self.current_mol.GetMol()))                                # Compute and return the standard QED value cast to a python float
        except Exception:                                                                   # Catch any exceptions raised by the QED heuristic module
            return 0.0                                                                      # Return absolute zero drug-likeness to safely absorb the failure

    def _calculate_reward(self) -> float:
        """
        Calculates the Multi-Parameter Objective (MPO) reward.
        
        Extracts physical descriptors (MW, logP, TPSA, HBD, HBA) and maps them through 
        sigmoid windows. Penalizes structural alerts and counts specific elements to counter 
        reward hacks. Combines an SA-like proxy score and complexity checks 
        (Bertz). Computes a geometric mean of all terms and scales it.
        
        Args:
            None.
            
        Returns:
            float: The heavily shaped and scaled terminal MPO reward.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> mpo_score = env._calculate_reward()
        """
        # ---------------------------------------------------------------------------------
        # Core Descriptor Extraction
        # Extract RDKit molecule representation and compute raw physical measurements.
        # ---------------------------------------------------------------------------------
        try:                                                                                # Wrap the entire multi-objective calculation to protect against catastrophic topology crashes
            mol = self.current_mol.GetMol()                                                 # Extract the standard RDKit Mol object from the RWMol (read-write) instance
            Chem.SanitizeMol(mol)                                                           # Sanitize the molecule to compute valences, ring information, and aromaticity

            qed = float(QED.qed(mol))                                                       # Quantitative Estimate of Drug-likeness (QED) 
            mw = float(Descriptors.MolWt(mol))                                              # Molecular weight (MW) of the molecule.
            logp = float(Crippen.MolLogP(mol))                                              # Octanol-water partition coefficient (cLogP), a measure of the molecule's hydrophobicity.
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))                                    # Topological Polar Surface Area (TPSA), which is a descriptor related to the molecule's ability to interact with polar solvents and is often used as a proxy for properties like solubility and permeability.
            hbd = int(Lipinski.NumHDonors(mol))                                             # Number of hydrogen bond donors (HBD) in the molecule (= number of atoms that can donate a hydrogen bond, e.g. -OH, -NH groups).
            hba = int(Lipinski.NumHAcceptors(mol))                                          # Number of hydrogen bond acceptors (HBA) in the molecule (= number of atoms that can accept a hydrogen bond, e.g. O, N atoms).
            rings = int(rdMolDescriptors.CalcNumRings(mol))                                 # Total number of rings in the molecule (indicator of molecular complexity and influences properties like rigidity and synthetic accessibility).
            bertz = float(GraphDescriptors.BertzCT(mol))                                    # Bertz complexity, a topological descriptor that quantifies the complexity of the molecule's structure based on its graph representation ---> higher values indicating more complex molecules.

            # -----------------------------------------------------------------------------
            # Structural Hacks & SA Proxy Features
            # Count elements and topology proxies to combat unstable policy loopholes.
            # -----------------------------------------------------------------------------
            hal = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")) # Count the number of halogen atoms (F, Cl, Br, I) in the molecule.
            p_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "P")                # Count the number of phosphorus atoms in the molecule.
            rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))                          # Number of rotatable bonds, which is a measure of the molecule's flexibility and can influence properties like binding affinity and bioavailability.
            heavy = int(mol.GetNumHeavyAtoms())                                             # Number of heavy atoms (non-hydrogen atoms) in the molecule, a simple measure of molecular size and complexity.
            arom_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))                    # Number of aromatic rings in the molecule, for properties like stability, reactivity, and interactions with biological targets.
            sp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))                             # Fraction of sp3-hybridized carbons with respect to total carbon count, a measure of the molecule's three-dimensionality and can influence properties like solubility and metabolic stability.
            bridge = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))                      # Number of bridgehead atoms (=part of two or more rings), can indicate the presence of fused or bridged ring systems, contributing to molecular complexity.
            spiro = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))                            # Number of spiro atoms (shared by two rings), can indicate the presence of spirocyclic structures, which are often found in bioactive molecules and can contribute to unique three-dimensional shapes.
            s_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "S")                # Sulfur count used to reduce sulfur-stacking loopholes.

            # -----------------------------------------------------------------------------
            # Target Windows Evaluation
            # Push raw values through sigmoid bounds to obtain [0,1] desirability scores.
            # -----------------------------------------------------------------------------
            d_mw = _sigmoid_window(mw, 180.0, 520.0)                                        # Evaluate weight viability window
            d_logp = _sigmoid_window(logp, 0.5, 4.5)                                        # Evaluate lipophilicity viability window
            d_tpsa = _sigmoid_window(tpsa, 20.0, 120.0)                                     # Evaluate topological polar surface area window
            d_hbd = _sigmoid_window(hbd, 0.0, 5.0)                                          # Evaluate hydrogen bond donor count window
            d_hba = _sigmoid_window(hba, 0.0, 10.0)                                         # Evaluate hydrogen bond acceptor count window
            d_rings = _sigmoid_window(rings, 1.0, 6.0)                                      # Evaluate general ring count window
            d_bertz = 1.0 / (1.0 + math.exp(0.004 * (bertz - 900.0)))                       # Evaluate generic complexity using a single sided decay


            d_hal = _sigmoid_window(float(hal), 0.0, 1.0)                                   # tightened to <=1 halogen to avoid halogen collapse in practice
            d_rot = _sigmoid_window(float(rot), 0.0, 8.0)                                   # "SA-like" proxy: prefers moderate size, moderate rotatable bonds, moderate aromaticity, not-too-flat
            d_heavy = _sigmoid_window(float(heavy), 15.0, 45.0)                             # Evaluate heavy atom limits window
            d_arom = _sigmoid_window(float(arom_rings), 0.0, 4.0)                           # Evaluate aromaticity limits window
            d_sp3 = _sigmoid_window(float(sp3), 0.15, 0.85)                                 # Evaluate fractional sp3 carbon window
            d_bridge = _sigmoid_window(float(bridge), 0.0, 2.0)                             # Evaluate bridged atom complexity limit window
            d_spiro = _sigmoid_window(float(spiro), 0.0, 2.0)                               # Evaluate spiro atom complexity limit window

            d_sulfur = _sigmoid_window(float(s_count), 0.0, 2.0)                            # Sulfur penalty term to reduce repeated sulfur stacking reward hacks (soft penalty; mask also enforces a hard cap)
            
            # -----------------------------------------------------------------------------
            # Proxy Aggregation & Penalties
            # Compute geometric means of subsets and apply hard chemical penalties.
            # -----------------------------------------------------------------------------
            
            # Synthetic Accessibility Proxy: Aggregate the individual SA-like proxy components (rotatable bonds, heavy atoms, aromatic rings, 
            # sp3 fraction, bridgehead atoms, spiro atoms) into a single synthetic accessibility proxy score using the geometric mean.
            d_sa_proxy = float(np.exp(np.mean(np.log(np.clip([d_rot, d_heavy, d_arom, d_sp3, d_bridge, d_spiro], 1e-6, 1.0))))) # np.clip to ensure that the individual components are between 1e-6 and 1.0 (not zero).
            
            p_pen = math.exp(-1.25 * float(p_count))                                        # Kept for safety if seeding ever introduces P, even though P is removed from action space

            alerts = _count_alerts(mol)                                                     # Retrieve the integer count of exact SMARTS toxicity alert matches
            alert_pen = math.exp(-1.0 * alerts)                                             # 1.0 if none, downweights if alerts exist
            # Gather all the metrics that will be used for the final MPO score
            components = [max(qed, 1e-6), d_mw, d_logp, d_tpsa, d_hbd, d_hba, d_rings, d_bertz, alert_pen, d_hal, d_sa_proxy, p_pen, d_sulfur] # Combine (geometric mean-ish)
            # Overall multi-parameter objective (MPO) score as the geometric mean of the individual component scores
            mpo = float(np.exp(np.mean(np.log(np.clip(components, 1e-6, 1.0)))))            # which include drug-likeness (QED), property desirability windows (MW, cLogP, TPSA, HBD/HBA, ring count, Bertz complexity), structural alerts penalty, and additional realism factors (halogen content, synthetic accessibility proxy, phosphorus penalty).

            return 12.0 * mpo                                                               # Scale the finalized MPO multiplier to inflate the numerical gradients
        except Exception:                                                                   # Catch any uncaught exception raised across the entirety of RDKit descriptors
            return 0.0                                                                      # Yield absolute zero to softly absorb calculation failure

    # -------- Chemistry constraints --------
    def _atom_max_valence(self, atom: Chem.Atom) -> int:
        """
        Determines a conservative heuristic for the maximum valence of an element.
        
        Inspects the string symbol of the provided atom and returns an integer ceiling 
        for how many bonds it can reasonably support without invoking deep hypervalency 
        errors.
        
        Args:
            atom (Chem.Atom): The RDKit atom to analyze.
            
        Returns:
            int: The maximum allowed valency.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> atom = Chem.Atom("O")
            >>> env._atom_max_valence(atom)
            2
        """
        sym = atom.GetSymbol()                                                              # Extract the string symbol for the query atom
        if sym == "C":                                                                      # Conservative typical valences (not exhaustive)
            return 4                                                                        # Return Carbon capacity
        if sym == "N":                                                                      # Check Nitrogen symbol
            return 3                                                                        # Return Nitrogen capacity
        if sym == "O":                                                                      # Check Oxygen symbol
            return 2                                                                        # Return Oxygen capacity
        if sym in ("F", "Cl", "Br", "I"):                                                   # Check if the element falls within the halogen group
            return 1                                                                        # Return minimal Halogen capacity
        if sym == "S":                                                                      # Check Sulfur symbol
            return 6                                                                        # allows hypervalence, still constrained by sanitize
        if sym == "P":                                                                      # Check Phosphorus symbol
            return 5                                                                        # Return Phosphorus capacity
        return 4                                                                            # Fallback constraint for unexpected elements

    def _atom_can_accept(self, atom_idx: int, bond_type: Chem.BondType) -> bool:
        """
        Evaluates whether an existing atom can legally form a new bond of a given type.
        
        Queries the current explicit valence of the atom and compares the remaining capacity 
        (max valence - current valence) against the bond order required by the new bond type.
        
        Args:
            atom_idx (int): The integer index of the atom to check.
            bond_type (Chem.BondType): The RDKit bond type requested (SINGLE, DOUBLE, etc).
            
        Returns:
            bool: True if the atom has enough free valency, False otherwise.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> env._atom_can_accept(0, Chem.BondType.DOUBLE)
            True
        """
        # Extract the RDKit atom object corresponding to the provided index from the current molecule, its maximum valence and current explicit valence. 
        # Then, determine if the atom can accept the new bond type based on these values, returning True if it can and False if it cannot.
        atom = self.current_mol.GetAtomWithIdx(int(atom_idx))                               # Retrieve the target RDKit atom object directly via its integer index
        max_v = self._atom_max_valence(atom)                                                # Retrieve the upper limit of valency allowed for this specific element
        try:                                                                                # Enter a try block because explicit valence can throw if molecule graph is deeply broken
            v = atom.GetExplicitValence()                                                   # returns the number of bonds the atom currently has
        except Exception:                                                                   # Catch the exception gracefully
            v = 0                                                                           # Fallback to zero current usage if RDKit cannot resolve explicit bonds
        required = 1 if bond_type == Chem.BondType.SINGLE else (2 if bond_type == Chem.BondType.DOUBLE else 3) # Map the abstract RDKit BondType object to its exact integer valence requirement
        return (max_v - v) >= required                                                      # Returns True if the available valence is >= to the required valence for the new bond (atom can accept the new bond). Otherwise, it returns False (violates valence rules for that atom).

    def _can_add(self, bond_type: Chem.BondType) -> bool:
        """
        Validates if adding a new atom with a specified bond to the focal node is allowed.
        
        Checks the global molecule size limit against `max_atoms` and verifies that the 
        currently focused atom can accept the required bond order via `_atom_can_accept`.
        
        Args:
            bond_type (Chem.BondType): The requested incoming bond order.
            
        Returns:
            bool: True if the operation conforms to size and valency limits, False otherwise.
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> env._can_add(Chem.BondType.SINGLE)
            True
        """
        if self.current_mol.GetNumAtoms() >= self.spec.max_atoms:                           # size cap
            return False                                                                    # Return false since the graph cannot accommodate any more atoms
        return self._atom_can_accept(int(self.focus_node_idx), bond_type)                   # True only if the focus atom can accept the new bond type (valence constraints) and the total number of atoms in the molecule is less than the maximum allowed (max_atoms). 

    def _get_observation(self):
        """
        Converts the RDKit molecule state into a PyTorch Geometric (PyG) graph representation.
        
        Iterates over all atoms to extract node features (one-hot element, hybridization, 
        aromaticity, ring membership, and a binary focus flag). Then iterates over 
        all bonds to build edge indices (undirected, so symmetric) and edge features 
        (one-hot bond order). Packages these into PyTorch tensors.
        
        Args:
            None.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A 4-tuple of 
            (node_features `x`, `edge_index`, `edge_attr`, and a dummy `node_mask`).
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> x, ei, ea, nm = env.reset()
            >>> x.shape
            torch.Size([1, 12])
        """
        # ---------------------------------------------------------------------------------
        # Graph Observation Generation
        # Map chemical states to normalized neural network feature vectors.
        # ---------------------------------------------------------------------------------
        assert self.current_mol is not None                                                 # Assert that the molecule is instantiated before attempting to extract features
        mol = self.current_mol                                                              # Bind the current read-write molecule to a local variable for easier access

        atom_features: List[List[float]] = []                                               # Initialize an empty list to store the computed feature vectors for all atoms
        
        # Iterate over each atom in the molecule to compute its feature vector, which includes one-hot encoding of the atom type, 
        # hybridization state, aromaticity, ring membership, and whether it is the currently focused node.
        for i, atom in enumerate(mol.GetAtoms()):                                           # Loop over each atom in the molecule alongside its positional index
            symbol = atom.GetSymbol()                                                       # Retrieve the string symbol (e.g., 'C', 'N') of the current atom
            type_feat = [1.0 if symbol == k else 0.0 for k in self.atom_types]              # One-hot encoding of the atom type based on the predefined list of allowed atom types in the ActionSpec.

            hyb = atom.GetHybridization()                                                   # Retrieve the hybridization state of the atom from RDKit
            hyb_feat = [1.0 if hyb == t else 0.0 for t in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)] # One-hot encode the hybridization state (SP, SP2, SP3) into a float list

            atom_features.append(type_feat + hyb_feat + [1.0 if atom.GetIsAromatic() else 0.0, 1.0 if atom.IsInRing() else 0.0, 1.0 if i == int(self.focus_node_idx) else 0.0,]) # Concatenate atom type, hybridization, aromaticity, ring membership, and focus flag into a single feature vector and append it

        x = torch.tensor(atom_features, dtype=torch.float32, device=self.device)            # shape [num_atoms, num_node_features=len(type_feat)+len(hyb_feat)+3]

        # Loop over all existing RDKit bonds globally within the molecule and build edge indices and edge features, 
        # ensuring to add both directions for undirected graph representation.
        rows, cols, attr = [], [], []                                                       # Initialize empty python lists to accumulate bidirectional graph connections and their features
        for bond in mol.GetBonds():                                                         # Loop over all existing RDKit bonds globally within the molecule
            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()                             # Unpack the source (begin) and target (end) atom indices for the current bond
            rows.extend([s, e])                                                             # Add both directions for undirected graph representation (source and target node indices for each edge).
            cols.extend([e, s])                                                             # Append the reversed order to the column tracking list to maintain undirected symmetry
            bt = bond.GetBondType()                                                         # Extract the abstract RDKit bond type object
            f = [1.0 if bt == t else 0.0 for t in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)] # Map the bond type to a discrete 4-class one-hot float encoding
            attr.extend([f, f])                                                             # undirected edges, so add both directions with the same features

        # ---------------------------------------------------------------------------------
        # Tensor Conversion & Edge Case Handling
        # Handle single-node graphs gracefully and build final PyG structures.
        # ---------------------------------------------------------------------------------
        if len(rows) == 0:                                                                  # Handle case with no bonds (e.g., single atom) to avoid empty tensors
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)          # Generate an explicitly empty 2D connectivity coordinate tensor
            edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)        # Generate an explicitly empty edge attribute tensor spanning 4 feature dimensions
        else:                                                                               # Execute if the graph possesses at least one valid bond
            # Edge index: shape [2, num_edges], where the first row contains source node indices and the second row contains target node indices for each edge in the molecule. 
            edge_index = torch.tensor([rows, cols], dtype=torch.long, device=self.device)   
            # Edge attributes: shape [num_edges, num_edge_features=4], where each row corresponds to the one-hot encoding of the bond type for that edge (single, double, triple, aromatic).
            edge_attr = torch.tensor(attr, dtype=torch.float32, device=self.device)         

        return x, edge_index, edge_attr, torch.zeros(x.size(0), dtype=torch.long, device=self.device)   # node_mask (not needed here since we don't pad, but can be used in future if we implement padding to max_atoms)

    # -------- Convenience --------
    def get_smiles(self) -> str:
        """
        Retrieves the canonical SMILES string for the current graph state.
        
        Attempts to translate the RDKit object to a SMILES format string. Catches any 
        conversion failures returning "INVALID" to prevent logging crashes.
        
        Args:
            None.
            
        Returns:
            str: The SMILES sequence or "INVALID".
            
        Example:
            >>> env = MoleculeEnvironment(torch.device("cpu"))
            >>> env.reset()
            >>> env.get_smiles()
            'C'
        """
        try:                                                                                # Wrap translation function to prevent logging-induced runtime crashes
            return Chem.MolToSmiles(self.current_mol.GetMol())                              # Process the underlying pure Mol object into a string sequence and return
        except Exception:                                                                   # Catch any internal serialization errors thrown by RDKit
            return "INVALID"                                                                # Yield a safe fallback string identifier indicating structural failure