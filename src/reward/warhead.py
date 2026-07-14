"""
=======================================
Covalent Warhead Reward Term (Stage 3).
=======================================

This script defines a specialized, warhead-aware reward function designed specifically 
for covalent KRAS G12C inhibitor generation. Because KRAS G12C inhibitors must bind 
covalently (typically via an electrophilic "warhead" undergoing conjugate addition 
to the Cys12 thiol), this module encodes a medicinal-chemistry prior directly into 
the reinforcement learning reward signal.

How it works:
It pre-compiles a list of SMARTS patterns representing classic Michael acceptors 
(e.g., acrylamides, vinyl sulfonamides). During the agent's generation trajectory, 
it scans the proposed 2D molecular graph. If a valid warhead is detected, it yields 
a small, bounded, additive reward bonus. This nudges the policy to "design-in" 
covalent-capable molecules organically, rather than relying on post-hoc filtering, 
without overwhelming the primary affinity, property, and diversity objectives.
"""

from rdkit import Chem

# -----------------------------------------------------------------------------------------
# Covalent Warhead SMARTS Definitions
# Defines structural patterns for Michael-acceptor / acrylamide-like electrophiles.
# These standard covalent warheads are pre-compiled into RDKit Mol objects for fast matching.
# -----------------------------------------------------------------------------------------
_WARHEAD_SMARTS = [                     # Initialize a list containing the SMARTS string representations of recognized covalent warheads
    "C=CC(=O)N",                        # Define acrylamide pattern (the primary warhead class used in sotorasib/adagrasib)
    "C=CC(=O)O",                        # Define acrylate ester or acid pattern (another standard Michael acceptor)
    "C=CC(=O)[#6]",                     # Define vinyl ketone pattern (carbon-bound alpha,beta-unsaturated carbonyl)
    "C=CS(=O)(=O)N",                    # Define vinyl sulfonamide pattern (sulfur-based Michael acceptor)
    "C#CC(=O)N",                        # Define propiolamide pattern (an alkyne-based Michael acceptor)
    "[CH2]=[CH]C(=O)",                  # Define generic enone pattern (terminal alkene conjugated with any carbonyl)
]                                                                                   
_WARHEAD_PATTERNS = [Chem.MolFromSmarts(s) for s in _WARHEAD_SMARTS]                # Pre-compile every SMARTS string into an RDKit Mol object to ensure efficient substructure searching during training


def has_warhead(mol) -> bool:
    """
    Evaluates whether the given molecule contains any recognized covalent warhead structure.
    
    Safely intercepts None-type invalid molecules. Then, it iterates through the globally 
    pre-compiled `_WARHEAD_PATTERNS`. Using RDKit's `HasSubstructMatch`, it checks the 
    molecular graph against these patterns, returning True upon the first successful match.
    
    Args:
        mol (Chem.Mol): The RDKit molecule object to be evaluated.
        
    Returns:
        bool: True if at least one covalent warhead motif is found, False otherwise.
        
    Example:
        >>> m = Chem.MolFromSmiles("C=CC(=O)NCC1=CC=CC=C1") # N-benzylacrylamide
        >>> has_warhead(m)
        True
    """
    # ---------------------------------------------------------------------------------
    # Warhead Substructure Matching
    # Safely iterates over the pre-compiled patterns to find any topological match.
    # ---------------------------------------------------------------------------------
    if mol is None:                                                                 # Check if the input molecule is None (e.g., due to RDKit parsing failure or invalid actions)
        return False                                                                # Return False immediately to prevent execution errors on broken chemical graphs
    # Iterate through compiled patterns and return True if the molecule contains at least one exact substructure match
    return any(p is not None and mol.HasSubstructMatch(p) for p in _WARHEAD_PATTERNS) 


def warhead_bonus(mol, bonus: float = 0.15) -> float:
    """
    Computes a bounded, additive reward scalar to incentivize covalent warhead generation.
        
    Calls `has_warhead(mol)`. If a warhead is verified, it returns the exact `bonus` value.
    This value is intentionally kept small so it functions as a gentle medicinal-chemistry 
    bias, nudging the policy toward covalent structures without dominating the overall MPO 
    (Multi-Parameter Objective).
    
    Args:
        mol (Chem.Mol): The RDKit molecule object to evaluate.
        bonus (float, optional): The fixed additive reward value. Defaults to 0.15.
        
    Returns:
        float: The exact float `bonus` if a warhead is present, otherwise exactly 0.0.
        
    Example:
        >>> m = Chem.MolFromSmiles("C=CC(=O)NCC1=CC=CC=C1") 
        >>> warhead_bonus(m)
        0.15
    """
    # ---------------------------------------------------------------------------------
    # Reward Bonus Evaluation
    # Resolves the logical check into a strictly formatted floating point reward scalar.
    # ---------------------------------------------------------------------------------
    return float(bonus) if has_warhead(mol) else 0.0                                # Evaluate the presence of a warhead and yield the specified float bonus if present, or exactly 0.0 otherwise