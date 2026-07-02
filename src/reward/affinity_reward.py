"""
===================================
Affinity Reward Calculation Module
===================================

This script isolates the mathematical transformations used to convert raw predicted binding 
affinities (and their associated predictive uncertainties) into a stable, well-behaved 
reward scalar for the reinforcement learning agent. By keeping this logic decoupled from 
the main multi-parameter objective (MPO) combiner, it maintains modularity, allows for 
independent unit testing, and enables easy swapping of scoring functions.

The pipeline utilizes a surrogate model ensemble to predict target binding affinity 
(z-scored pChEMBL). This module takes that raw z-score and squashes it into a (0, 1) 
range using a sigmoid function. Crucially, it combats "proxy-hacking" (where the RL agent 
exploits blind spots in the surrogate model by generating bizarre, out-of-distribution graphs) 
by heavily penalizing the reward if the ensemble's predictions disagree (high uncertainty). 
The agent is thus forced to find molecules that are not only predicted to be strong binders, 
but are also within the confident domain of the predictive oracle.
"""

import math

def normalise_affinity(aff_hat_z: float) -> float:
    """
    Transforms a z-scored predicted pChEMBL value into a bounded (0, 1) continuous scalar.
    
    Applies the standard logistic sigmoid function. A z-score of 0 (mean affinity) maps 
    exactly to 0.5. A positive z-score (+1 standard deviation) maps to ~0.73, while a 
    negative z-score (-1 standard deviation) maps to ~0.27. This provides a smooth, 
    differentiable gradient for the reinforcement learning agent to climb.
    
    Args:
        aff_hat_z (float): The z-scored predicted affinity (e.g., pChEMBL) from the surrogate.
        
    Returns:
        float: The normalized affinity score strictly bounded between 0.0 and 1.0.
        
    Example:
        >>> normalise_affinity(0.0)
        0.5
        >>> round(normalise_affinity(1.0), 2)
        0.73
    """
    # -----------------------------------------------------------------------------------------
    # Sigmoid Transformation
    # Map the unbounded continuous z-score securely into the (0, 1) probability space.
    # -----------------------------------------------------------------------------------------
    return 1.0 / (1.0 + math.exp(-float(aff_hat_z)))                                        # Apply the standard logistic sigmoid function to map the unbounded z-score smoothly into the (0, 1) range.


def affinity_term(aff_hat_z: float, aff_unc_z: float, beta: float) -> float:
    """
    Calculates the final, robust affinity reward (A=sigmoid(predicted_mean_affinity)-β*std) 
    by penalizing uncertain predictions.
    
    First, it normalizes the raw predicted affinity using the `normalise_affinity` function. 
    Then, it subtracts an uncertainty penalty. This penalty is the product of the raw 
    uncertainty (ensemble's std) z-score (`aff_unc_z`) and a scaling hyperparameter (`beta`). 
    Finally, because the subtraction might push the result below 0 (or technically above 1 if 
    uncertainty is negative), it explicitly clamps the final output to [0, 1].
    
    Args:
        aff_hat_z (float): The z-scored predicted affinity from the surrogate model.
        aff_unc_z (float): The z-scored uncertainty metric (e.g., ensemble variance).
        beta (float): The hyperparameter controlling the strength of the uncertainty penalty.
        
    Returns:
        float: The confident, penalized affinity reward score clamped between 0.0 and 1.0.
        
    Example:
        >>> # High affinity, low uncertainty
        >>> round(affinity_term(2.0, -1.0, 0.1), 2)
        0.98
        >>> # High affinity, but highly uncertain (penalty applied)
        >>> round(affinity_term(2.0, 3.0, 0.2), 2)
        0.28
    """
    # -----------------------------------------------------------------------------------------
    # Uncertainty Penalization & Clamping
    # Calculate the base score (normalized predicted mean affinity), subtract the safety 
    # penalty (β*ensemble's_std), and clamp to [0, 1].
    # -----------------------------------------------------------------------------------------
    a = normalise_affinity(aff_hat_z) - beta * float(aff_unc_z)                             # Compute the base normalized affinity and subtract the weighted uncertainty penalty to discourage proxy-hacking.
    return max(0.0, min(1.0, a))                                                            # Clamp the penalized score strictly between 0.0 and 1.0 to ensure a stable reward scalar for the RL agent.