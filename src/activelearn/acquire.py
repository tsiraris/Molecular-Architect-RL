"""
===================================
Active-Learning Acquisition "Brain"
===================================

This script (src/activelearn/acquire.py) forms the core of the active-learning decision engine. 
Its primary role is to decide exactly which generated molecules should be evaluated by the 
computationally expensive docking oracle during Stage 3. Returns the list of indices to dock.

It combines two critical concepts to maximize information gain per oracle call:
  1. Acquisition Score (UCB): Uses an Upper Confidence Bound score defined as 
     a(x) = mu(x) + kappa * sigma(x). This elegantly balances exploitation (high predicted 
     affinity `mu` from the surrogate ensemble mean) against exploration (high epistemic 
     uncertainty `sigma` from the ensemble standard deviation). The `kappa` parameter is 
     annealed from high to low across rounds: it explores the surrogate's blind spots 
     early, and strictly exploits a trusted surrogate late in the process.
  2. Diverse Batch Selection: If we only took the top scores, the batch would be saturated 
     with near-duplicate analogues, wasting expensive oracle calls. Instead, this module 
     clusters a larger pool of top candidates by Tanimoto similarity (using the Butina 
     algorithm) and selects one representative per cluster round-robin, maximizing 
     chemical information per dock.

Everything here is engineered in pure Python + RDKit + NumPy (no GPU dependency), 
ensuring it is lightweight and strictly unit-testable offline.
"""
from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def ucb(mean_z: np.ndarray, unc_z: np.ndarray, kappa: float) -> np.ndarray:
    """
    Computes the Upper Confidence Bound (UCB) acquisition score for a batch of molecules.
    
    Combines the predicted mean affinity (exploitation) and the predicted uncertainty 
    (exploration) scaled by the kappa factor.
    
    Args:
        mean_z (np.ndarray): Array of predicted mean affinity scores from the ensemble.
        unc_z (np.ndarray): Array of epistemic uncertainties (standard deviations).
        kappa (float): The current exploration-exploitation trade-off parameter.
        
    Returns:
        np.ndarray: An array of UCB scores used to rank the candidates.
    """
    return np.asarray(mean_z) + kappa * np.asarray(unc_z)                                   # Compute array of UCB scores scaling uncertainty by kappa and adding to mean


def kappa_schedule(round_idx: int, n_rounds: int, k_start: float = 2.0, k_end: float = 0.2) -> float:
    """
    Computes the annealed kappa value for the current active learning round.
    
    Linearly interpolates kappa from a highly exploratory starting value (k_start) down 
    to a greedy, exploitative ending value (k_end) as the active learning rounds progress.
    
    Args:
        round_idx (int): The current active learning round index (0-indexed).
        n_rounds (int): The total number of planned active learning rounds.
        k_start (float, optional): The initial high-exploration kappa. Defaults to 2.0.
        k_end (float, optional): The final high-exploitation kappa. Defaults to 0.2.
        
    Returns:
        float: The calculated kappa multiplier for the current round.
    """
    if n_rounds <= 1:                                                                       # If only one round exists, default directly to the greedy end state
        return k_end                                                                        # Return the final exploitation-focused kappa value
    t = round_idx / (n_rounds - 1)                                                          # Calculate the normalized progress fraction between 0.0 and 1.0
    return (1 - t) * k_start + t * k_end                                                    # Return linearly interpolated kappa from start to end


def _fp(smi):
    """
    Internal helper to generate a Morgan fingerprint from a SMILES string.
    
    Args:
        smi (str): The input SMILES string.
        
    Returns:
        ExplicitBitVect or None: A 2048-bit Morgan fingerprint (radius 2), or None if parsing fails.
    """
    m = Chem.MolFromSmiles(smi)                                                                 # Parse the raw SMILES string into an RDKit Mol object
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None   # Compute 2048-bit Morgan fingerprint (radius 2) if parsed successfully, else return None


def butina_pick(smiles: List[str], n_pick: int, cutoff: float = 0.65) -> List[int]:
    """
    Selects a structurally diverse subset of molecules using Butina clustering.
    
    Converts SMILES to Morgan fingerprints, computes a pairwise Tanimoto distance matrix, 
    and applies Butina clustering at the specified cutoff. It then iterates round-robin 
    over the sorted clusters (largest first), picking the cluster centroid first, followed 
    by subsequent members, until `n_pick` candidates are gathered.
    
    Args:
        smiles (List[str]): List of candidate SMILES strings to diversify.
        n_pick (int): The exact number of diverse representatives to select.
        cutoff (float, optional): The Tanimoto similarity cutoff for clustering. Defaults to 0.65.
        
    Returns:
        List[int]: A list of integer indices mapping back to the input `smiles` list.
    """
    # Convert all top candidates SMILES to Morgan fingerprints, 
    # create a list of valid ones and store total valid count. 
    fps = [_fp(s) for s in smiles]                                                          # Convert all input SMILES strings into their respective Morgan fingerprints
    valid = [i for i, f in enumerate(fps) if f is not None]                                 # Extract indices of SMILES that successfully produced a valid fingerprint
    if not valid:                                                                           # Check if the entire list failed to parse into valid fingerprints
        return []                                                                           # Return an empty list since no clustering can be performed
    vfps = [fps[i] for i in valid]                                                          # Create a filtered list containing strictly the valid fingerprints
    n = len(vfps)                                                                           # Store the total count of valid fingerprints to cluster
    # Initialize an empty list to store the condensed distance matrix (1 - Tanimoto_similarity).
    dists = []                                                                              
    # Compute the condensed distance matrix for Butina clustering (1 - similarity of the fingerprint and all others).
    for i in range(1, n):                                                                   # Loop over the valid fingerprints starting from the second element
        sims = DataStructs.BulkTanimotoSimilarity(vfps[i], vfps[:i])                        # Compute Tanimoto similarity between current fingerprint and all previous ones
        dists.extend(1.0 - s for s in sims)                                                 # Convert similarities to distances (1 - sim) and append to the condensed matrix
    # Perform Butina clustering using the distance matrix and a distance cutoff (1 - sim_cutoff)
    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)                  
    clusters = sorted(clusters, key=len, reverse=True)                                      # Sort the resulting clusters by size in descending order (largest clusters first)
    picks, ci = [], 0                                                                       # Initialize the list for selected indices and a counter for the round-robin cluster selection
    # Round-Robin selection: Iterate over clusters, picking one representative 
    # from each in turn until we have enough picks or all clusters are exhausted.
    while len(picks) < n_pick and any(clusters):                                            # Loop until we have enough picks or all clusters are completely exhausted
        c = clusters[ci % len(clusters)]                                                    # Select the next cluster using modulo arithmetic for round-robin cycling
        if c:                                                                               # Check if the currently selected cluster still has unpicked members
            picks.append(valid[c[0]]); clusters[ci % len(clusters)] = c[1:]                 # Append global index of cluster centroid (first element), then remove it from the cluster
        ci += 1                                                                             # Increment the round-robin counter to move to the next cluster on the next iteration
        # If all clusters are empty, break early to avoid infinite looping
        if all(len(c) == 0 for c in clusters):                                              # Check if every single cluster has been completely emptied of members
            break                                                                           # Terminate the selection loop early if no more candidates exist
        # Return exactly the requested number of diverse representatives (n_pick), 
        # each from a different cluster, until all clusters are exhausted.
        return picks[:n_pick]                                                               


def select_batch(smiles: List[str], mean_z: np.ndarray, unc_z: np.ndarray,
                 n_dock: int, round_idx: int, n_rounds: int,
                 pool_mult: int = 4, cutoff: float = 0.65) -> List[int]:
    """
    Executes the full active learning acquisition pipeline.
    
    Computes the annealed kappa, ranks all molecules by their UCB score, and extracts an 
    oversampled top pool (e.g., 4x the target dock count). It then passes this highly-ranked 
    pool into the Butina clustering algorithm to select a structurally diverse, high-value 
    batch for docking.
    
    Args:
        smiles (List[str]): Full list of generated SMILES strings.
        mean_z (np.ndarray): Ensemble mean affinity predictions for all SMILES.
        unc_z (np.ndarray): Ensemble uncertainty predictions for all SMILES.
        n_dock (int): The final number of molecules to send to the docking oracle.
        round_idx (int): Current active learning round index.
        n_rounds (int): Total active learning rounds planned.
        pool_mult (int, optional): Multiplier to create the pre-clustering pool. Defaults to 4.
        cutoff (float, optional): Butina clustering Tanimoto cutoff. Defaults to 0.65.
        
    Returns:
        List[int]: A list of global indices representing the final diversified batch to dock.
    """
    # Compute the annealed kappa value for the current round
    k = kappa_schedule(round_idx, n_rounds)                                                 # Compute the specific exploration-exploitation trade-off parameter for this round
    # Calculate the Upper Confidence Bound (UCB) scores for all candidates and rank them
    scores = ucb(mean_z, unc_z, k)                                                          # Calculate the Upper Confidence Bound scores for all candidates using predictions and uncertainty
    order = np.argsort(-scores)                                                             # Sort indices to rank candidates by their UCB scores in descending order (highest first)
    # Form a candidate pool by slicing the top-ranked candidates, sized by a 
    # multiple of the target dock count, and extract their corresponding SMILES strings.
    pool = list(order[: max(n_dock * pool_mult, n_dock)])                                   # Slice the top candidates to form a candidate pool, sized by a multiple of the target dock count
    pool_smiles = [smiles[i] for i in pool]                                                 # Extract the actual SMILES strings corresponding to the top-ranked candidate indices
    # Cluster the top pool and select a diverse subset of local indices using Butina clustering
    local = butina_pick(pool_smiles, n_dock, cutoff)                                        
    # Map the selected local pool indices back to their original global indices 
    # and return them as the final list of indices to be sent to the docking oracle.
    return [pool[i] for i in local]                                                         