"""
============================================
Standalone 2D Property Objective Calculator
============================================

This script isolates the logic for computing the already established 2D property objective: 
Quantitative Estimate of Drug-likeness (QED) + a small multi-parameter objective (MPO).
While `chem_env` computes this term  (_calculate_reward / MPO + the QED curriculum) during 
active Phase-1 training, this standalone module is maintained for downstream Stage-2 processes, 
such as (a) unit tests, (b) offline scoring of arbitrary SMILES, and (c) readers who want the 
property maths in one place without reading the env. 

The script provides a continuous scoring function (`property_mpo`) that evaluates a molecule 
based on ideal drug-like windows for Molecular Weight (MW) and Lipophilicity (logP), alongside 
its raw QED score. Crucially, it aggregates these via a geometric mean, ensuring that a molecule
must be well-rounded to score highly; a terrible score in one descriptor severely penalizes the 
final output, preventing trivial property hacking.
"""

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen

# -------------------------------------------------------------------------------------------------
# Bounding & Scaling Heuristics
# Mathematical utility function to map raw physical values into bounded [0, 1] desirability scores.
# -------------------------------------------------------------------------------------------------
def _window(x: float, lo: float, hi: float) -> float:
    """
    Computes a smooth [0, 1] desirability score based on linear decay outside a target window.
    
    Returns exactly 1.0 if the target value `x` falls inclusively between `lo` and `hi`. 
    If `x` is outside this window, the score decays linearly to 0.0 over a distance 
    equal to the width of the window itself (`hi - lo`). This provides a forgiving gradient 
    for reinforcement learning algorithms rather than a harsh boolean cliff.
    
    Args:
        x (float): The raw molecular property value being evaluated.
        lo (float): The lower bound of the ideal property window.
        hi (float): The upper bound of the ideal property window.
        
    Returns:
        float: A normalized desirability scalar strictly between 0.0 and 1.0.
        
    Example:
        >>> _window(300.0, 250.0, 500.0)
        1.0
        >>> _window(200.0, 250.0, 500.0) # Decays linearly below 250
        0.8
    """
    if x < lo:                                                                              # Check if the value falls below the target window's lower bound
        return max(0.0, 1.0 - (lo - x) / (hi - lo))                                         # Calculate linear penalty down to zero based on distance from the lower bound
    if x > hi:                                                                              # Check if the value exceeds the target window's upper bound
        return max(0.0, 1.0 - (x - hi) / (hi - lo))                                         # Calculate linear penalty down to zero based on distance from the upper bound
    return 1.0                                                                              # Return maximum score since the value is perfectly inside the target window


# -----------------------------------------------------------------------------------------
# Multi-Parameter Objective Evaluation (MPO)
# Core metric aggregation blending independent chemical descriptors into a single scalar.
# -----------------------------------------------------------------------------------------
def property_mpo(mol: Chem.Mol) -> float:
    """
    Calculates a geometric-mean multi-property objective (MPO) score comprised only of 
    QED, Molecular Weight (MW), and the Lipophilicity logarithm (LogP).
        
    Extracts Molecular Weight, LogP, and QED from the provided RDKit molecule. 
    It applies the `_window` function to constrain MW (ideal: 250-500) and logP 
    (ideal: 1.0-4.0). It computes the geometric mean of these bounded properties 
    plus raw QED. The geometric mean heavily penalizes candidates that fail completely 
    on even a single axis.
    
    Args:
        mol (Chem.Mol): The instantiated RDKit molecule to be evaluated.
        
    Returns:
        float: The final aggregated MPO score, safely bounded [0.0, 1.0].
        
    Example:
        >>> m = Chem.MolFromSmiles("CC1=CC=CC=C1") # Toluene
        >>> score = property_mpo(m)
        >>> type(score)
        <class 'float'>
    """
    if mol is None:                                                                         # Safely handle empty inputs to prevent crashes during automated batch scoring
        return 0.0                                                                          # Return an absolute zero score for non-existent or failed molecule parses
    try:                                                                                    # Wrap RDKit descriptor calculations to catch deep chemical graph parsing errors
        mw = Descriptors.MolWt(mol)                                                         # Compute the exact physical molecular weight of the graph
        logp = Crippen.MolLogP(mol)                                                         # Estimate the octanol-water partition coefficient (lipophilicity)
        qed = QED.qed(mol)                                                                  # Compute the standard Quantitative Estimate of Drug-likeness score
    except Exception:                                                                       # Catch any unexpected failures thrown by the underlying RDKit C++ backend
        return 0.0                                                                          # Fallback to a zero score gracefully without interrupting the evaluation pipeline
    
    terms = [_window(mw, 250, 500), _window(logp, 1.0, 4.0), qed]                           # Package the bounded desirability scores into an aggregation list
    prod = 1.0                                                                              # Initialize a running product accumulator for the geometric mean calculation
    for t in terms:                                                                         # Iterate sequentially over every individual objective term in the list
        prod *= max(1e-6, t)                                                                # Multiply the accumulator by the term, clamping strictly above zero to avoid mathematical collapse
    # The returned MPO score is the nth root of the geometric mean of the bounded descriptors 
    # (windowed MW, LogP, and raw QED), so that if a molecule fails on even a single axis, 
    # its final score is penalized severely.
    return prod ** (1.0 / len(terms))                                                       # Return the nth root of the product, finalizing the geometric mean computation


# -----------------------------------------------------------------------------------------
# Safely Wrapped Heuristics
# Isolated fallback functions for strictly fetching descriptors without crashing the run.
# -----------------------------------------------------------------------------------------
def safe_qed(mol: Chem.Mol) -> float:
    """
    Computes the Quantitative Estimate of Drug-likeness (QED) safely.
        
    Wraps the standard RDKit `QED.qed()` call in a try-except block. Used primarily 
    during early training curriculum steps where the agent generates unstable, 
    highly strained topological graphs that frequently crash standard analytical tools.
    
    Args:
        mol (Chem.Mol): The RDKit molecule to be analyzed.
        
    Returns:
        float: The QED score [0.0, 1.0], or 0.0 if the calculation throws an exception.
        
    Example:
        >>> m = Chem.MolFromSmiles("CCO")
        >>> safe_qed(m) > 0.3
        True
    """
    if mol is None:                                                                         # Verify the input molecule object actually exists before passing to RDKit
        return 0.0                                                                          # Return strictly zero drug-likeness for null inputs
    try:                                                                                    # Encapsulate the RDKit QED call to intercept topological analysis exceptions
        return float(QED.qed(mol))                                                          # Compute the QED value and explicitly cast it to a native python float scalar
    except Exception:                                                                       # Catch all potential internal calculation errors triggered by strange chemistry
        return 0.0                                                                          # Return zero safely to penalize the broken structure without crashing the loop