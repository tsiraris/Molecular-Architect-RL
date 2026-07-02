"""
====================================================
Stage-2 Terminal-Reward Combiner & Diversity Archive
====================================================

This module defines the Stage-2 reward system, which upgrades the Phase-1 generative baseline 
into a target-aware, synthesizability-gated design engine. 

While the Stage-1 reward (gated x curriculum-blended property MPO) is kept intact in `chem_env`, 
Stage-2 wraps that property term (P) with two new signals: Affinity (A) and Diversity (D).

The overall reward calculation follows this mathematical formulation:
    total_unit = w_property * P 
               + (w_affinity * curriculum_ratio) * A      # Affinity ramps in with the curriculum
               - w_diversity * D                          # Penalizes similarity (anti mode-collapse) between the generated molecule and the archive (rollout molecules)
    
    reward     = reward_scale * total_unit * soft         # soft = synthesizability-gate multiplier

Component Definitions:
    P : Property term in ~[0, 1]. This is the chem_env's gated/curriculum base, divided by its ceiling.
    A : Affinity term in ~[0, 1]. Calculated as `normalise_affinity(aff_hat_z) - beta * aff_unc_z` (clamped >= 0).
    D : Diversity penalty in [0, 1]. The mean Tanimoto similarity of the molecule to a rolling archive.

Backward Compatibility:
    If `use_affinity=False` and `use_diversity=False`, then `w_property=1.0` and `reward_scale=12.0`. 
    This results in: reward = 12 * (base/12) * soft = base * soft (the exact Stage-1 reward).

Anti-Hacking Guards (By Design):
    1. The Synthesizability Gate (`soft`): Stops descriptor exploits (e.g., S#C / cumulene / high SA).
    2. The Uncertainty Penalty (`beta * aff_unc`): Stops *surrogate* exploits. It heavily penalizes 
       adversarial inputs where the proxy is wrongly optimistic because those points exhibit high 
       ensemble disagreement.
    3. The Diversity Penalty (`D`): Directly attacks the mode collapse measured in the Stage-1 run 
       (which suffered from 0.02 uniqueness). A molecule identical to what the policy keeps emitting 
       is now strictly worth less.
"""

from collections import deque
from typing import Deque, Dict, Optional, Tuple

from rdkit.Chem import AllChem, DataStructs

from .affinity_reward import normalise_affinity, affinity_term


class DiversityArchive:
    """
    Rolling Morgan-fingerprint archive shared across the vectorized environments.
    
    Maintains a bounded double-ended queue (deque) of the most recently generated molecules' 
    Morgan fingerprints. Because the queue is bounded, it keeps the comparison local-in-time 
    and guarantees O(1) maintenance overhead. It exposes a penalty function that computes 
    the mean Tanimoto similarity of a new molecule against this recent history.
    
    Args:
        maxlen (int, optional): Maximum capacity of the rolling archive. Defaults to 256.
        radius (int, optional): Morgan fingerprint radius. Defaults to 2.
        nbits (int, optional): Morgan fingerprint bit vector length. Defaults to 1024.
        
    Example:
        >>> archive = DiversityArchive(maxlen=100)
        >>> archive.add(Chem.MolFromSmiles("CCO"))
        >>> p = archive.penalty(Chem.MolFromSmiles("CCO"))
        >>> print(p) # Expected to be 1.0 (identical)
    """
    def __init__(self, maxlen: int = 256, radius: int = 2, nbits: int = 1024):
        # -----------------------------------------------------------------------------------------
        # Archive State Initialization
        # Set up the rolling queue and store fingerprint algorithmic parameters.
        # -----------------------------------------------------------------------------------------
        self.fps: Deque = deque(maxlen=maxlen)                                                      # Initialize a bounded deque to automatically age out old fingerprints, preserving O(1) appends
        self.radius = radius; self.nbits = nbits                                                    # Store the Morgan fingerprint calculation parameters (radius and bit length) on a single line

    def _fp(self, mol):
        """
        Internal helper to safely compute the Morgan fingerprint for a given RDKit molecule.
        
        Wraps the RDKit fingerprint generation in a try-except block to gracefully handle 
        chemically invalid or null molecules without crashing the training loop.
        
        Args:
            mol (Chem.Mol): The RDKit molecule object to process.
            
        Returns:
            DataStructs.ExplicitBitVect or None: The calculated bit vector, or None if failed.
        """
        # -----------------------------------------------------------------------------------------
        # Fingerprint Computation
        # -----------------------------------------------------------------------------------------
        try:                                                                                        # Wrap the external C++ RDKit call to catch deep topological compilation errors
            return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.nbits)              # Compute and return the circular Morgan fingerprint bit vector using stored settings
        except Exception:                                                                           # Catch any exception raised during the RDKit processing phase
            return None                                                                             # Return None safely to indicate a failed feature extraction

    def penalty(self, mol) -> float:
        """
        Calculates the diversity penalty for a newly generated molecule.
        
        Computes the fingerprint of the provided molecule and calculates its bulk Tanimoto 
        similarity against the entire current rolling archive. Returns the average similarity.
        0.0 means completely novel (or empty archive), 1.0 means identical to the archive average.
        
        Args:
            mol (Chem.Mol): The generated RDKit molecule to evaluate.
            
        Returns:
            float: The mean Tanimoto similarity penalty in the range [0.0, 1.0].
        """
        # -----------------------------------------------------------------------------------------
        # Penalty Evaluation
        # Compare a new molecule against the historical distribution to quantify mode collapse.
        # -----------------------------------------------------------------------------------------
        if mol is None or len(self.fps) == 0:                                                       # Check if the input is null or if the historical archive hasn't been populated yet
            return 0.0                                                                              # Return zero penalty since there is nothing to compare against or the molecule is invalid
        fp = self._fp(mol)                                                                          # Attempt to extract the bit vector fingerprint for the current molecule
        if fp is None:                                                                              # Check if the fingerprint extraction failed due to graph invalidity
            return 0.0                                                                              # Return zero penalty to avoid penalizing structurally invalid molecules twice
        # Compute the pairwise Tanimoto similarity against every fingerprint currently in the archive
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.fps))                               
        return float(sum(sims) / len(sims)) if sims else 0.0                                        # Calculate and return the mean similarity, defaulting to 0.0 if the similarity list is unexpectedly empty

    def add(self, mol):
        """
        Ingests a new molecule into the rolling diversity archive.
        
        Extracts the fingerprint and appends it to the right side of the deque. 
        If the deque is full, the oldest fingerprint on the left is automatically discarded.
        
        Args:
            mol (Chem.Mol): The molecule to add to the historical record.
        """
        # -----------------------------------------------------------------------------------------
        # Archive Appending
        # -----------------------------------------------------------------------------------------
        if mol is None:                                                                             # Verify that the provided molecule object actually exists before processing
            return                                                                                  # Exit early without mutating the archive if the molecule is null
        fp = self._fp(mol)                                                                          # Safely calculate the Morgan fingerprint bit vector for the valid molecule
        if fp is not None:                                                                          # Confirm that the fingerprint extraction was mathematically successful
            self.fps.append(fp)                                                                     # Append the new fingerprint to the bounded deque, automatically evicting the oldest item if full


def default_reward_cfg() -> Dict:
    """
    Provides the default hyperparameter configuration for the Stage-2 reward system.
        
    Returns a dictionary of weights and toggles. The default values are explicitly tuned 
    to exactly reproduce the Stage-1 behavior (affinity and diversity turned off).
    
    Returns:
        Dict: Configuration key-value pairs governing reward aggregation.
        
    Example:
        >>> cfg = default_reward_cfg()
        >>> print(cfg["use_affinity"])
        False
    """
    # -----------------------------------------------------------------------------------------
    # Default Configuration Factory
    # -----------------------------------------------------------------------------------------
    return {                                                                                        # Begin returning the statically defined default configuration dictionary
        "use_affinity": False,                                                                      # Disable the target-aware affinity oracle scoring by default for backward compatibility
        "use_diversity": False,                                                                     # Disable the Tanimoto similarity mode-collapse penalty by default
        "w_property": 1.0,                                                                          # Set the fundamental weight of the base 2D MPO property objective to fully active (1.0)
        "w_affinity": 1.0,                                                                          # Set the fundamental weight of the affinity term (only applies if use_affinity is True)
        "w_diversity": 0.0,                                                                         # Zero out the diversity penalty multiplier explicitly
        "beta_uncertainty": 0.5,                                                                    # Define the subtraction scalar (beta) for penalizing high ensemble disagreement in the proxy
        "reward_scale": 12.0,                                                                       # Define the final global multiplier that expands the normalized [0,1] reward into numerical gradients
        "property_ceiling": 12.0,                                                                   # chem_env's MPO base ceiling, used to normalise P into ~[0,1]
    }                                                                                               


def combine(base: float, soft: float, curriculum_ratio: float, cfg: Dict,
            aff_hat_z: Optional[float] = None, aff_unc_z: Optional[float] = None,
            diversity_pen: float = 0.0) -> Tuple[float, Dict]:
    """
    Aggregates the gated property base with affinity and diversity signals into a single reward.
    
    Steps:
    1. Normalizes the base property reward (P) using the configured ceiling.
    2. Calculates the Affinity (A) term (normalized prediction minus scaled uncertainty) if active.
    3. Fetches the Diversity (D) penalty if active.
    4. Computes the `total_unit` as a weighted linear combination of P, A, and D.
       Crucially, the affinity term scales in dynamically with the `curriculum_ratio`.
    5. Multiplies the total unit by the global scale and the strict synthesizability gate (`soft`).
    
    *Reminder: The overall reward calculation is:
    total_unit = w_property * P 
               + (w_affinity * curriculum_ratio) * A      # Affinity ramps in with the curriculum
               - w_diversity * D                          # Penalizes similarity (anti mode-collapse) between the generated molecule and the archive (rollout molecules)
    
    reward     = reward_scale * total_unit * soft         # soft = synthesizability-gate multiplier

    Component Definitions:
        P : Property term in ~[0, 1]. This is the chem_env's gated/curriculum base, divided by its ceiling.
        A : Affinity term in ~[0, 1]. Calculated as `normalise_affinity(aff_hat_z) - beta * aff_unc_z` (clamped >= 0).
        D : Diversity penalty in [0, 1]. The mean Tanimoto similarity of the molecule to a rolling archive.
    
    Args:
        base (float): chem_env's curriculum-blended property reward (pre-gate-multiplier).
        soft (float): The synthesizability gate multiplier in [0,1].
        curriculum_ratio (float): The chronological 0->1 training schedule progression scalar.
        cfg (Dict): The active reward tuning configuration dictionary.
        aff_hat_z (Optional[float]): Z-scored predicted affinity from the oracle proxy.
        aff_unc_z (Optional[float]): Z-scored ensemble uncertainty from the oracle proxy.
        diversity_pen (float, optional): Pre-computed Tanimoto similarity penalty. Defaults to 0.0.
        
    Returns:
        Tuple[float, Dict]: The final aggregated scalar reward, alongside a tracking dict of components (P, A, D, mean affinity, std).
        
    Example:
        >>> cfg = default_reward_cfg()
        >>> reward, info = combine(base=6.0, soft=1.0, curriculum_ratio=0.5, cfg=cfg)
        >>> print(reward) # 12.0 * (1.0 * (6.0/12.0)) * 1.0 = 6.0
        6.0
    """
   
    # Scale the raw base property (P) reward score into a standardized [0, 1] regime.
    P = base / max(1e-8, cfg["property_ceiling"])                                                   # Calculate P by dividing the raw base score by the configured ceiling, safeguarding against division by zero
    A = 0.0                                                                                         # Initialize the Affinity component strictly to zero as a fallback
    
    # If affinity and diversity terms are enabled, based on configuration flags, compute them.
    if cfg["use_affinity"] and aff_hat_z is not None:                                               # Check if the affinity oracle is enabled and valid predictions were provided
        A = affinity_term(aff_hat_z, aff_unc_z if aff_unc_z is not None else 0.0, cfg["beta_uncertainty"]) # Compute the uncertainty-penalized affinity term to neutralize surrogate exploitation
    D = diversity_pen if cfg["use_diversity"] else 0.0                                              # Bind the active diversity penalty if configured, otherwise zero it out to ignore mode collapse

    # Aggregation & Final Scaling: Linearly combine the components and apply the rigid synthesizability gate.
    total_unit = (cfg["w_property"] * P                                                             # Multiply the normalized property term by its defined configuration weight
                  + (cfg["w_affinity"] * curriculum_ratio) * A                                      # Ramp the weighted affinity term smoothly into the objective using the curriculum progression scalar
                  - cfg["w_diversity"] * D)                                                         # Subtract the weighted diversity penalty to aggressively punish redundant topological generation
    reward = cfg["reward_scale"] * total_unit * soft                                                # Compute the final reward by multiplying the scaled total unit by the continuous synthesizability gate
    
    info = {"P": P, "A": A, "D": D,                                                                 # Package the normalized component scalars into an analytical tracking dictionary
            "aff_hat_z": aff_hat_z, "aff_unc_z": aff_unc_z}                                         # Append the raw z-scored oracle predictions and uncertainties to the tracking dictionary
    
    return float(reward), info                                                                      # Return the rigorously cast final floating-point reward alongside the breakdown dictionary