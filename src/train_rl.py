"""
=================================================
Proximal Policy Optimization (PPO) Training Loop
=================================================

This script serves as the primary entry point for training the Molecular Architect agent. 
It integrates the environment (`VectorMoleculeEnv`), the actor-critic network (`MoleculeAgent`), 
and the experience buffer (`PPOBuffer`) into a cohesive reinforcement learning loop.

The script employs a synchronous, vectorized PPO algorithm to optimize the agent. 
Multiple environments are run in parallel to gather diverse experiences efficiently. 
The training uses Generalized Advantage Estimation (GAE) for stable policy updates, 
Automatic Mixed Precision (AMP) for GPU acceleration, and a dynamic linear curriculum 
that transitions the reward landscape from simple heuristics (QED) to a stringent 
Multi-Parameter Objective (MPO). Extensive logging (WandB and text files) tracks 
structural diversity, property distributions, and best-found molecules.

Stage-2 (Target-Aware) Upgrades:
This updated loop includes conditional wiring for the Stage-2 target-aware design phase. 
It can integrate a pre-trained `AffinityScorer` deep ensemble surrogate, a `DiversityArchive` 
to penalize mode collapse via rolling Tanimoto checks, FiLM-based protein pocket conditioning 
via pre-computed embeddings, and real-time Synthetic Accessibility (SA) score tracking.
"""

from __future__ import annotations
import datetime as dt
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim
import wandb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from chem_env import ActionSpec
from gnn_agent import MoleculeAgent
from ppo import PPOBuffer
from vec_env import VectorMoleculeEnv
import os as _os, sys as _sys
from rdkit.Chem import RDConfig as _RDConfig
_sys.path.append(_os.path.join(_RDConfig.RDContribDir, "SA_Score"))
import sascorer as _sascorer

# -----------------------------------------------------------------------------------------
# Global Configuration & Setup
# Suppress RDKit logs and define the primary hyperparameters for the RL training run.
# -----------------------------------------------------------------------------------------
RDLogger.DisableLog("rdApp.*")                                                              # Disable excessive RDKit C++ backend warnings to keep the console output clean

CONFIG = {                                                                                  # Initialize the global configuration dictionary holding all hyperparameters
    # Optimization parameters
    "lr": 3e-4,                                                                             # Set the learning rate for the AdamW optimizer to update network weights
    "gamma": 0.99,                                                                          # Set the discount factor for future rewards in Markov Decision Process
    "gae_lambda": 0.95,                                                                     # Set the lambda parameter for Generalized Advantage Estimation smoothing
    "clip_epsilon": 0.2,                                                                    # Set the PPO clipping threshold to prevent destructively large policy updates
    "target_kl": 0.02,                                                                      # Set the target KL divergence threshold to trigger early stopping in an epoch
    "entropy_coef_start": 0.08,                                                             # Set the initial exploration bonus multiplier to encourage random early actions
    "entropy_coef_end": 0.02,                                                               # Set the final minimal exploration bonus multiplier for late-stage exploitation
    "value_coef": 0.5,                                                                      # Set the weighting factor for the critic's value loss in the total loss sum
    "max_grad_norm": 1.0,                                                                   # Set the maximum allowed norm for gradients to prevent exploding gradients
    # Training parameters
    "batch_size": 256,                                                                      # Set the number of transitions processed simultaneously during a PPO update
    "ppo_epochs": 10,                                                                       # Set the number of times the network iterates over the collected buffer data
    "buffer_size": 2048,                                                                    # Set total transitions per update (across envs) to gather before updating
    "num_envs": 8,                                                                          # Set the number of parallel vectorized environments to run concurrently
    "updates": 300,                                                                         # Set the total number of data collection and PPO update cycles to execute
    # Model parameters
    "hidden_dim": 128,                                                                      # Set the dimensionality of the hidden feature vectors in the GNN layers
    "edge_dim": 4,                                                                          # Set the dimensionality of the one-hot encoded bond type edge features
    # Env/action parameters
    "max_atoms": 25,                                                                        # Set the absolute maximum number of atoms permitted in a generated molecule
    "max_steps": 40,                                                                        # Set the maximum number of structural edit actions permitted per episode
    "min_atoms": 5,                                                                         # Set the minimum required atoms before the agent is allowed to stop safely
    # Curriculum (terminal reward mix) parameters
    "curriculum_start": 25,                                                                 # Set the update iteration where the reward begins shifting from QED to MPO
    "curriculum_end": 300,                                                                  # Set the update iteration where the reward strictly becomes the full MPO
    # Eval/logging parameters
    "log_every": 5,                                                                         # Set the frequency (in updates) at which metrics are evaluated and logged
    "save_every": 50,                                                                       # Set the frequency (in updates) at which model weights and top-k are saved
    "eval_samples": 256,                                                                    # Set the molecules per evaluation snapshot for computing diversity metrics
    "project": "molecular-rl-2026",                                                         # Set the Weights & Biases project name to organize remote tracking dashboards
    "run_name": None,                                                                       # Initialize the specific run identifier, allowing auto-generation if None
    "seed": 42,                                                                             # Set the master random seed ensuring deterministic, reproducible initializations
    # --------------------------------------------------------------------------------------------------------------
    # Stage-2 Configuration (Target-aware knobs). 
    # Defaults below reproduce the Stage-1 run exactly (affinity and diversity off). 
    # Flip use_affinity/use_diversity on for the Stage-2 experiment.
    # --------------------------------------------------------------------------------------------------------------
"use_affinity": True,                                                                      # Add the learned-surrogate affinity term to the terminal reward
    "surrogate_dir": "../artifacts/surrogate_kras",                                         # Directory holding the trained deep ensemble + norm.json
    "w_property": 0.5,                                                                      # Weight on the 2D property term P (kept as a low prior when affinity is on)
    "w_affinity": 0.7,                                                                      # Weight on the affinity term A (ramped by the curriculum_ratio)
    "beta_uncertainty": 0.5,                                                                # Penalty coefficient on ensemble uncertainty (anti proxy-hacking)
    "reward_scale": 12.0,                                                                   # Overall scale so reward magnitude matches Stage-1 (base ceiling was ~12)
    # Add the Tanimoto-to-archive diversity penalty (anti mode-collapse)
    "use_diversity": True,                                                                  
    # Weight on the diversity penalty D (~0.5–1.0 to fight collapse)
    "w_diversity": 1.0,                                                                      
    # Rolling window of recent molecules used for the diversity penalty
    "diversity_archive_size": 256,                                                           
    # Enable pocket-FiLM conditioning of the policy (single-target KRAS)
    "use_pocket": True,                                                                    
    # Path to the pocket embedding produced by pocket/encode_pocket.py
    "pocket_npy": "../data/kras_g12c_pocket.npy",                                           
}                                                                                           

def set_seed(seed: int) -> None:
    """
    Sets the random seeds across all numerical libraries for reproducibility.
    
    How it works:
    Calls the seeding functions for Python's built-in `random`, `numpy`, and `torch` 
    (both CPU and CUDA) to ensure stochastic operations yield identical results across runs.
    
    Args:
        seed (int): The integer value to use as the universal random seed.
        
    Returns:
        None
        
    Example:
        >>> set_seed(42)
    """
    
    # Random Seeding: Lock the pseudo-random number generators across all libraries.
    random.seed(seed)                                                                       # Fix the seed for Python's native random module operations
    np.random.seed(seed)                                                                    # Fix the seed for Numpy's numerical random generation routines
    torch.manual_seed(seed)                                                                 # Fix the seed for PyTorch's CPU-bound tensor random generations
    torch.cuda.manual_seed_all(seed)                                                        # Fix the seed for PyTorch's GPU-bound (CUDA) tensor random generations


def tanimoto_diversity(smiles: List[str]) -> Tuple[float, float, float]:
    """
    Calculates validity, uniqueness, and Morgan-fingerprint Tanimoto diversity for a batch.
    
    Parses SMILES to RDKit molecules. Filters out invalid parses. Computes Morgan 
    fingerprints (radius 2, 1024 bits) for valid items. Averages the pairwise Tanimoto 
    similarity across the batch and subtracts from 1.0 to yield a diversity metric.
    
    Args:
        smiles (List[str]): A list of SMILES string representations.
        
    Returns:
        Tuple[float, float, float]: A tuple containing (valid_rate, unique_rate, diversity).
        
    Example:
        >>> batch = ["C", "CC", "INVALID", "C"]
        >>> val, uniq, div = tanimoto_diversity(batch)
    """
    # ------------------------------------------------------------------------------------------------------
    # Diversity Metric Calculation: Convert SMILES to fingerprints to statistically analyze batch variation.
    # ------------------------------------------------------------------------------------------------------
    mols = [Chem.MolFromSmiles(s) for s in smiles]                                          # Attempt to parse every string in the provided list into an RDKit Mol object
    valid_idx = [i for i, m in enumerate(mols) if m is not None]                            # Identify and collect the indices of molecules that successfully compiled
    if not valid_idx:                                                                       # Check if the list of valid molecule indices is entirely empty
        return 0.0, 0.0, 0.0                                                                # If no valid molecules exist, safely return zeros for all diversity metrics

    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=1024) for i in valid_idx]# Compute Morgan fingerprints for valid molecules (radius 2, 1024 bits) for similarity calculation
    sim = 0.0                                                                               # Initialize a running accumulator for the total pairwise Tanimoto similarity
    for i in range(len(fps)):                                                               # Iterate over every generated fingerprint via its index
        # Calculate similarities between the current fingerprint and the entire batch
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps)                                 
        # Add normalized similarities (sum(s) - 1.0 to exclude self-similarity) to accumulator
        sim += (sum(s) - 1.0) / (len(fps) - 1 + 1e-6)                                       

    # Compute the three core metrics: validity, uniqueness, and diversity
    uniq = len(set(smiles[i] for i in valid_idx)) / max(1, len(valid_idx))                  # Calculate uniqueness as the fraction of unique valid SMILES among all valid molecules
    valid_rate = len(valid_idx) / len(smiles)                                               # Calculate validity as the fraction of successfully parsed molecules over the total input batch
    diversity = 1.0 - (sim / (len(fps) + 1e-6))                                             # Calculate structural diversity as 1.0 minus the average batch Tanimoto similarity
    return valid_rate, uniq, diversity                                                      # Return the computed triad of generation quality metrics


def scaffold(smiles: str) -> str:
    """
    Extracts the Bemis-Murcko structural scaffold from a given molecule.

    Strips away side chains to isolate the central ring systems and linkers. 
    Useful for assessing structural novelty beyond mere functional group decoration.
    
    Args:
        smiles (str): The SMILES string of the query molecule.
        
    Returns:
        str: The SMILES string of the isolated scaffold, "NONE", or "INVALID".
        
    Example:
        >>> scaffold("CC1=CC=CC=C1")
        'c1ccccc1'
    """
    # Bemis-Murcko Scaffold Extraction of a given SMILES string (strips side chains)
    m = Chem.MolFromSmiles(smiles)                                                          # Convert the input raw SMILES string into an RDKit topological molecule object
    if m is None:                                                                           # Check if the string translation resulted in a null object (parsing failure)
        return "INVALID"                                                                    # Return the designated error string if the molecule is chemically malformed
    try:                                                                                    # Wrap scaffold calculation in a try block to catch deep structural graph errors
        scaf = MurckoScaffold.GetScaffoldForMol(m)                                          # Invoke the RDKit Bemis-Murcko module to strip side-chains and retain core rings
        return Chem.MolToSmiles(scaf) if scaf is not None else "NONE"                       # Convert the core object back to string, or return "NONE" if it is acyclic
    except Exception:                                                                       # Catch any underlying algorithmic failure during graph traversal
        return "INVALID"                                                                    # Return the designated error string as a safe fallback


def mol_props(smiles: str) -> Dict[str, float]:
    """
    Calculates standard molecular descriptors for logging purposes.

    Translates the SMILES to an RDKit Mol and computes QED, MolWt, LogP, TPSA, 
    Hydrogen Bond Donors/Acceptors, and Ring Counts. 
    
    Args:
        smiles (str): The string representation of the molecule.
        
    Returns:
        Dict[str, float]: A dictionary containing key-value pairs of the descriptors.
        
    Example:
        >>> props = mol_props("c1ccccc1")
        >>> props["rings"]
        1.0
    """
    
    # Descriptor Profiling: Parse the SMILES string, convert to RDKit, and extract scalar physicochemical features.
    m = Chem.MolFromSmiles(smiles)                                                          # Parse the target SMILES string back into an active RDKit graph object
    if m is None:                                                                           # Check if the compilation failed due to valency or syntax errors
        return {"valid": 0.0}                                                               # Yield a dictionary with a zeroed validity flag and omit other descriptors
    return {                                                                                # Begin constructing the comprehensive descriptor output dictionary
        "valid": 1.0,                                                                       # Hardcode the validity flag to 1.0 since parsing succeeded
        "qed": float(QED.qed(m)),                                                           # Calculate and store the Quantitative Estimate of Drug-likeness
        "mw": float(Descriptors.MolWt(m)),                                                  # Calculate and store the precise physical molecular weight
        "logp": float(Crippen.MolLogP(m)),                                                  # Calculate and store the predicted octanol-water partition coefficient
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),                                        # Calculate and store the topological polar surface area
        "hbd": float(Lipinski.NumHDonors(m)),                                               # Calculate and store the discrete count of hydrogen bond donors
        "hba": float(Lipinski.NumHAcceptors(m)),                                            # Calculate and store the discrete count of hydrogen bond acceptors
        "rings": float(rdMolDescriptors.CalcNumRings(m)),                                   # Calculate and store the total enumerated count of topological rings
    }                                                                                       # Close the property dictionary block

def _snapshot_sa(smiles_list):
    """
    Calculates the average Synthetic Accessibility (SA) score for a batch of molecules.
    
    Iterates over a provided list of SMILES strings, attempting to parse each into an 
    RDKit molecule object. For successfully parsed molecules, it invokes the external 
    `_sascorer` module to calculate their individual SA scores (typically ranging from 
    1 to 10, where 1 is easy to synthesize and 10 is very difficult). Finally, it computes 
    and returns the arithmetic mean SA score of the valid batch.
    
    Args:
        smiles_list (List[str]): A list of SMILES string representations to evaluate.
        
    Returns:
        float: The mean SA score of all valid molecules in the batch, or NaN (Not a Number) 
        if no valid molecules could be parsed or scored successfully.
        
    Example:
        >>> batch = ["CCO", "c1ccccc1", "INVALID_SMILES"]
        >>> avg_sa = _snapshot_sa(batch)
        >>> type(avg_sa)
        <class 'float'>
    """
    vals = []                                                                                   # Initialize an empty list to accumulate the successfully computed SA scores

    # Loop through the batch, parse SMILES strings into RDKit molecule objects, 
    # and compute individual SA scores for the valid ones.
    for s in smiles_list:                                                                       # Iterate sequentially through each SMILES string provided in the input list
        m = Chem.MolFromSmiles(s)                                                               # Attempt to parse the raw string sequence into a valid RDKit topological graph
        if m is not None:                                                                       # Check if the string successfully compiled into a graph without valency errors
            try:                                                                                # Wrap the scoring function in a try block to intercept complex heuristic failures
                vals.append(_sascorer.calculateScore(m))                                        # Calculate the SA score using the fragment-based heuristic and append to list
            except Exception:                                                                   # Catch any arbitrary exceptions thrown by the underlying C++ or Python scorer
                pass                                                                            # Silently ignore the error and proceed to the next molecule in the batch

    # Compute the batch average, returning a safe NaN if the entire batch was invalid.
    return float(np.mean(vals)) if vals else float("nan")


class TopK:
    """
    Maintains a sorted list of the highest-reward molecules discovered during training.
    
    Internally keeps a list of `(reward, smiles)` tuples. Each time a new candidate is added, 
    the list is sorted in descending order by reward, and truncated to `k` elements.
    """
    def __init__(self, k: int = 50):
        """
        Initializes the TopK tracker.
        
        Args:
            k (int, optional): The maximum number of elite molecules to retain. Defaults to 50.
        """
        # TopK Initialization
        self.k = k                                                                          # Store the user-defined maximum capacity constraint for the elite archive
        self.items: List[Tuple[float, str]] = []                                            # Initialize the empty python list designated to hold the reward-SMILES tuples

    def add(self, reward: float, smiles: str):
        """
        Evaluates and potentially inserts a new candidate into the elite archive.
        
        Args:
            reward (float): The final terminal reward achieved.
            smiles (str): The corresponding SMILES sequence.
        """
        # TopK Addition & Sorting: Insert new data, sort descending by reward, and enforce the capacity limit.
        if smiles is None:                                                                  # Check if the provided string is a null-type representing a fatal generation
            return                                                                          # Immediately exit the function without mutating the tracking list
        self.items.append((float(reward), smiles))                                          # Append the newly discovered candidate as a tuple to the tracking list
        self.items.sort(key=lambda x: x[0], reverse=True)                                   # Sort the entire list in-place dynamically based on the reward scalar descending
        self.items = self.items[: self.k]                                                   # Truncate the list to strictly enforce the configured upper capacity bound

    def best(self) -> Tuple[float, str]:
        """
        Retrieves the single best molecule found so far.
        
        Returns:
            Tuple[float, str]: The reward and SMILES string of the top entry.
        """
        # TopK Retrieval of the Highest Reward Candidate: Return the first tuple in the sorted list, or a safe default if empty.
        if not self.items:                                                                  # Check if the tracking list remains completely unpopulated
            return 0.0, "None"                                                              # Return a safe zero-reward default to prevent indexing errors on empty lists
        r, s = self.items[0]                                                                # Unpack the first tuple (guaranteed highest reward due to sorting)
        return r, s                                                                         # Return the highest historical reward and its string structure

    def to_rows(self) -> List[Dict]:
        """
        Converts the stored archive into a list of detailed dictionaries for tabular saving.
        
        Returns:
            List[Dict]: A list of dicts containing rewards, strings, and full properties.
        """
        # Map the TopK list of simple tuples into deep property dictionaries for the final report.
        rows = []                                                                           # Initialize an empty list to gather the deeply processed data dictionaries
        for r, s in self.items:                                                             # Iterate sequentially through the stored elite candidate tuples
            p = mol_props(s)                                                                # Compute the full suite of RDKit descriptors for the candidate string
            rows.append({"reward": r, "smiles": s, **p, "scaffold": scaffold(s)})           # Package reward, string, unpacked properties, and scaffold into a dict and append
        return rows                                                                         # Return the fully processed list of data dictionaries


def entropy_coef(update: int) -> float:
    """
    Calculates the dynamic entropy coefficient using a linear decay schedule.
    It calculates a multiplier (coefficient) that determines how much the agent is
    encouraged to "explore" (take random actions) versus "exploit" (take best-known actions).
    
    It divides training into three distinct phases based on the current update:
    - Phase 1: Maximum Exploration (t = 0.0): If update <= 25, the agent uses entropy_coef_start (e.g., 0.08).
    - Phase 2: Linear Decay (0.0 < t < 1.0): If update is between 25 and 300, it calculates t, which is the percentage of progress through the transition window. It then blends the start and end values.
    - Phase 3: Maximum Exploitation (t = 1.0): If update >= 300, the agent locks into entropy_coef_end (e.g., 0.01).
    
    Interpolates between `entropy_coef_start` and `entropy_coef_end` based on the 
    current update step, governed by the curriculum bounds. This enforces exploration 
    early and exploitation late.
    
    Args:
        update (int): The current PPO update iteration index.
        
    Returns:
        float: The exact entropy coefficient multiplier to use for this step.
        
    Example:
        >>> # Assuming start=0.08, end=0.01, bounds=[25, 300]
        >>> entropy_coef(100)
        0.06090909090909091
    """
    # Entropy Schedule Computation: Linearly decay the exploration multiplier aligned with the reward curriculum bounds.
    u0, u1 = CONFIG["curriculum_start"], CONFIG["curriculum_end"]                           # Extract the chronological start and end anchors for the linear schedule
    t = 0.0 if update <= u0 else (1.0 if update >= u1 else (update - u0) / max(1, (u1 - u0))) # Calculate the normalized progress fraction strictly bounded between 0.0 and 1.0
    return (1.0 - t) * CONFIG["entropy_coef_start"] + t * CONFIG["entropy_coef_end"]        # Return the linearly interpolated entropy coefficient multiplier


def save_topk_txt(path: str, rows: List[Dict]) -> None:
    """
    Writes the top-k molecule dictionary list to disk as an aligned, human-readable ASCII table.
    
    Extracts all keys, enforces a preferred column order, calculates maximum string widths 
    per column, and writes formatted strings separated by vertical bars.
    
    Args:
        path (str): The destination file path.
        rows (List[Dict]): The expanded list of dictionaries from `TopK.to_rows()`.
    """
    # TopK Table File Writing: Determine column layouts, format floats, and export ASCII lines.
    if not rows:                                                                            # Terminate early if the input list of dictionaries is entirely empty
        return                                                                              # Safely exit the writing function to avoid errors on empty data
    os.makedirs(os.path.dirname(path), exist_ok=True)                                       # Create the necessary parent directories on disk, ignoring errors if they exist
    cols = list(rows[0].keys())                                                             # Dynamically extract the list of column names from the first dictionary

    preferred = ["reward", "smiles", "qed", "mw", "logp", "tpsa", "hbd", "hba", "rings", "valid", "scaffold"] # Define a human-readable hardcoded order for the primary columns
    cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]    # Merge the preferred order with any newly discovered miscellaneous keys

    def fmt(v):                                                                             # Define an internal helper function to coerce diverse types into clean strings
        if isinstance(v, float):                                                            # Check if the target variable is a floating point number
            return f"{v:.4g}"                                                               # Apply a compact 4-significant-digit format to floating point variables
        return str(v)                                                                       # Fallback to standard python string casting for integers and base strings

    widths = {c: max(len(c), max(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}      # Compute the absolute maximum character width required for every individual column
    sep = " | "                                                                             # Define the separator string to place between columns
    header = sep.join([c.ljust(widths[c]) for c in cols])                                   # Construct the primary header string by left-justifying column names
    rule = "-" * len(header)                                                                # Generate a horizontal divider line of hyphens matching the header length

    with open(path, "w", encoding="utf-8") as f:                                            # Open the target destination file in write mode using standard utf-8 encoding
        f.write(header + "\n")                                                              # Write the constructed column header string to the top of the file
        f.write(rule + "\n")                                                                # Write the horizontal hyphen divider line underneath the header
        for r in rows:                                                                      # Iterate sequentially through every dictionary representation in the payload
            line = sep.join([fmt(r.get(c, "")).ljust(widths[c]) for c in cols])             # Construct a left-justified row string matching the calculated column widths
            f.write(line + "\n")                                                            # Append the fully formatted tabular row string to the open text file


class ResearchLogger:
    """
    A custom logger that outputs a continuous textual record of the RL training process.
    
    Creates a timestamped text file, dumps the hyperparameter config header, and exposes a 
    `log_step` method to append heavily formatted numeric data rows during training.
    """
    def __init__(self, config: Dict, out_dir: str = "experiments"):
        """
        Initializes the logger and writes the configuration header to disk.
        
        Args:
            config (Dict): The full hyperparameter dictionary.
            out_dir (str, optional): The base folder for logs. Defaults to "experiments".
        """
        # Logger Initialization: Open file handler, write run metadata, and establish tabular headers.
        os.makedirs(out_dir, exist_ok=True)                                                 # Ensure the targeted base logging directory physically exists on the filesystem
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                                # Generate a precise string timestamp corresponding to the exact current time
        self.path = os.path.join(out_dir, f"Run_{ts}.txt")                                  # Construct the final absolute file path incorporating the timestamp

        with open(self.path, "w", encoding="utf-8") as f:                                   # Open the newly minted log file in strict write mode
            f.write("MOLECULAR RL (2026 baseline)\n" + "=" * 80 + "\n")                     # Inject a bold title block and divider into the top of the log file
            for k, v in config.items():                                                     # Iterate over the entire provided configuration key-value pairs
                f.write(f"{k:<24}: {v}\n")                                                  # Append each setting to the file using left-justified alignment
            f.write("=" * 80 + "\n")                                                        # Inject a secondary bold divider underneath the configuration block
            header = (                                                                      # Define the multi-line formatted string representing the tabular metrics header
                f"{'Update':<6} | {'R_mean':<8} | {'R_std':<8} | {'Valid%':<7} | {'Unique%':<8} | "     # Define layout bounds for step tracking and baseline rewards
                f"{'Div':<6} | {'KL':<8} | {'Clip':<6} | {'ExpVar':<7} | {'Ent':<6} | {'Curr':<6} | "   # Define layout bounds for RL diagnostic variables
                f"{'BestR':<7} | BestSMILES\n"                                              # Define layout bounds for elite tracking
            )                                                                               # Close header string declaration
            f.write(header)                                                                 # Append the fully resolved header layout string to the open file
            f.write("-" * (len(header) - 1) + "\n")                                         # Append a dynamically sized horizontal divider line mirroring the header width

    def log_step(
        self,
        update: int,
        r_mean: float,
        r_std: float,
        valid_rate: float,
        unique_rate: float,
        diversity: float,
        kl: float,
        clip_frac: float,
        exp_var: float,
        entropy_coef_val: float,
        curriculum_ratio: float,
        best_reward: float,
        best_smiles: str,
    ) -> None:
        """
        Appends a newly computed row of evaluation metrics to the ongoing log file.
        
        Args:
            update (int): Current iteration step.
            ... [various numeric training metrics]
            best_smiles (str): Best string found.
        """
        # Logger Row Appending: Truncate strings and append neatly aligned numeric rows.
        s_short = (best_smiles[:80] + "…") if len(best_smiles) > 80 else best_smiles        # Safely truncate excessively long SMILES strings to maintain table layout
        row = (                                                                             # Define the multi-line payload string integrating all current evaluation data
            f"{update:<6} | {r_mean:<8.3f} | {r_std:<8.3f} | {100*valid_rate:<7.1f} | {100*unique_rate:<8.1f} | " # Inject core metrics and percentages
            f"{diversity:<6.3f} | {kl:<8.4f} | {clip_frac:<6.3f} | {exp_var:<7.3f} | {entropy_coef_val:<6.3f} | " # Inject raw numeric diagnostic tracking variables
            f"{curriculum_ratio:<6.3f} | {best_reward:<7.3f} | {s_short}\n"                  # Inject curriculum state and best candidate snippet
        )                                                                                   # Close row string declaration
        with open(self.path, "a", encoding="utf-8") as f:                                   # Open the existing text log file in append mode
            f.write(row)                                                                    # Directly inject the assembled evaluation string as a new line


def train():
    """
    The main execution wrapper coordinating the Proximal Policy Optimization (PPO) loop.
    
    How it works:
    1. Initializes hardware bindings, random seeds, environments, networks, and buffers.
    2. Runs outer loops (`updates` times) to collect batches of molecular edits.
    3. Solves the Markov states, yielding actions, log probs, and simulated environment rewards.
    4. Bootstraps final values and uses Generalized Advantage Estimation (GAE) to compute returns.
    5. Optimizes the actor-critic network over multiple inner `ppo_epochs` using clipped surrogates.
    6. Employs Automatic Mixed Precision (AMP) scaling to accelerate backpropagation safely.
    7. Dispatches logging, TopK updates, and checkpoint saves periodically.
    
    Returns:
        None
    """
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Main Train Initialization
    # Lock seeds, bind compute hardware, and spawn core RL infrastructure objects such as the environment, agent, optimizer, buffer, and logger.
    # ------------------------------------------------------------------------------------------------------------------------------------------
    set_seed(CONFIG["seed"])                                                                # Set random seeds for reproducibility of the training process across random, numpy, and torch (both CPU and CUDA).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   # Query the hardware stack to bind the fastest available tensor processing device

    spec = ActionSpec(max_atoms=CONFIG["max_atoms"])                                        # Create an action specification for the environment to initialize the environment and determine the action space size.
    num_actions = spec.num_actions                                                          # Extract the scalar dimension of the total unmasked flat action array
    input_feats = len(spec.atom_types) + 3 + 3                                              # Compute node dimension: atom one-hot + hyb(3) + flags(3)

    run_name = CONFIG["run_name"] or f"molrl_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}" # Resolve the experiment identifier or auto-generate one based on time
    wandb.init(                                                                             # Initiate the connection to the Weights & Biases remote tracking server
        project=CONFIG["project"],                                                          # Bind the session to the overarching remote organizational project name
        name=run_name,                                                                      # Assign the specific human-readable title to this exact execution run
        config={**CONFIG, "num_actions": num_actions, "input_feats": input_feats},          # Upload the complete dictionary of hyperparameters and network dims to WandB
    )                                                                                       # Close the WandB initialization call

    # ----------------------------------------------------------------------------------------------------
    # Stage-2 Setup (Affinity, Diversity, FiLM Pocket Conditioning - All Optional) 
    # **Reminder: With the default CONFIG these are all no-ops (identical to the Stage-1 run).
    # ---------------------------------------------------------------------------------------------------
    
    # If enabled, build the affinity scorer and diversity archive
    affinity_scorer = None                                                                          # Initialize the proxy scorer placeholder to None by default
    if CONFIG["use_affinity"]:                                                                      # Check if the Stage-2 target-aware surrogate scoring flag is active
        from surrogate.predict import AffinityScorer                                                # Imported lazily so Stage-1 runs need no surrogate present
        affinity_scorer = AffinityScorer(CONFIG["surrogate_dir"], device=str(device))               # Load the trained deep ensemble + label-normalisation stats once
        print(f"[stage2] affinity surrogate loaded from {CONFIG['surrogate_dir']}")                 # Echo the successful surrogate mounting to the terminal
    
    diversity_archive = None                                                                        # Initialize the Tanimoto archive placeholder to None by default
    if CONFIG["use_diversity"]:                                                                     # Check if the Stage-2 rolling diversity penalty flag is active
        from reward.composite import DiversityArchive                                               # Shared rolling Tanimoto archive across all envs
        diversity_archive = DiversityArchive(maxlen=CONFIG["diversity_archive_size"])               # Instantiate the rolling buffer to track recently generated scaffolds
        print(f"[stage2] diversity penalty ON (archive={CONFIG['diversity_archive_size']}, w={CONFIG['w_diversity']})") # Echo the diversity penalty activation to the terminal
    
    # Assemble the reward-config dictionary for the env's composite combiner reads
    reward_cfg = {                                                                                  # Mirror of composite.default_reward_cfg(), populated from CONFIG
        "use_affinity": CONFIG["use_affinity"], "use_diversity": CONFIG["use_diversity"],           # Pass operational boolean flags down to the composite reward engine
        "w_property": CONFIG["w_property"], "w_affinity": CONFIG["w_affinity"],                     # Pass numerical scalar weights defining the linear objective combination
        "w_diversity": CONFIG["w_diversity"], "beta_uncertainty": CONFIG["beta_uncertainty"],       # Pass penalty coefficients for mode collapse and epistemic uncertainty
        "reward_scale": CONFIG["reward_scale"], "property_ceiling": 12.0,                           # Pass final multiplication scalars to match gradient magnitudes
    }                                                                                               # Close the reward configuration dictionary

    env = VectorMoleculeEnv(CONFIG["num_envs"], device, action_spec=spec,
                            affinity_scorer=affinity_scorer, diversity_archive=diversity_archive,
                            reward_cfg=reward_cfg)                                                  # Initialize a vectorized environment for molecular generation (parallel simulation of multiple environments/molecules simultaneously).
    
    # Also if enabled, load the pocket embedding
    pocket_dim = 0                                                                                  # Initialize the target pocket dimensionality fallback counter to zero
    pocket_vec = None                                                                               # Initialize the FiLM modulation vector placeholder to None by default
    if CONFIG["use_pocket"]:                                                                        # Check if the structural pocket conditioning flag is active
        import numpy as _np                                                                         # Import numpy strictly for loading the localized pre-computed embeddings
        pocket_vec = _np.load(CONFIG["pocket_npy"])                                                 # 1-D pocket embedding (ESM or deterministic fallback)
        pocket_dim = int(pocket_vec.shape[0])                                                       # Extract the exact feature length of the loaded pocket embedding vector
        print(f"[stage2] pocket conditioning ON (dim={pocket_dim}) from {CONFIG['pocket_npy']}")    # Echo the pocket FiLM network activation to the terminal
    
    agent = MoleculeAgent(input_feats, num_actions, hidden_dim=CONFIG["hidden_dim"], edge_dim=CONFIG["edge_dim"], pocket_dim=pocket_dim).to(device) # Initialize the actor-critic neural network and dispatch its weights to compute memory
    if pocket_vec is not None:                                                                      # Verify if a valid pocket embedding was successfully extracted
        agent.set_pocket(pocket_vec)                                                                # Install the fixed single-target pocket vector used by FiLM
    
    optimizer = optim.AdamW(agent.parameters(), lr=CONFIG["lr"])                            # Attach the AdamW gradient optimization algorithm to update the agent's tensor weights
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))                          # Initialize the gradient scaling engine to enable safe fp16 mixed precision
    buffer = PPOBuffer(device, num_envs=CONFIG["num_envs"], gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"]) # Allocate the monolithic memory buffer to store trajectory rollouts
    topk = TopK(k=50)                                                                       # Initialize the high-score tracking archive restricted to 50 molecules

    obs = env.reset()                                                                       # Reset the environment to get the initial observations for each parallel environment.

    out_dir = os.path.join("artifacts", run_name)                                           # Create an output directory for saving checkpoints and logs, organized under "artifacts" with a subdirectory named after the run.
    os.makedirs(out_dir, exist_ok=True)                                                     # Force create the directory structure ignoring any pre-existing warnings
    logger = ResearchLogger({**CONFIG, "num_actions": num_actions, "input_feats": input_feats}, out_dir=out_dir)    # Spin up the local textual backup tracking engine

    # -------------------------------------------------------------------------------------
    # Outer Update Loop
    # Governs dataset collection, value bootstrapping, and network gradient steps.
    # -------------------------------------------------------------------------------------
    for update in range(CONFIG["updates"]):                                                 # Iterate over the overarching master data collection and learning epochs
        # Compute the curriculum ratio based on the current update number, which determines how much the reward is influenced by the curriculum (intermediate rewards) versus the final reward.
        curr_ratio = min(1.0, max( 0.0, (update - CONFIG["curriculum_start"])/max(1, (CONFIG["curriculum_end"] - CONFIG["curriculum_start"])))) # Calculate 0 to 1 scalar

        buffer.clear()                                                                      # Purge all stale transition memory from the previous gradient update cycle

        # ---------------------------------------------------------------------------------
        # Environment Rollout (Collection Phase)
        # Advance the simulator sequentially, recording transitions and rewards into RAM.
        # ---------------------------------------------------------------------------------
        ep_term_rewards = []                                                                # Tracker for terminal rewards of episodes finished this update
        steps_per_env = CONFIG["buffer_size"] // CONFIG["num_envs"]                         # Compute the exact integer number of transitions to pull from each parallel worker
        for _ in range(steps_per_env):                                                      # Iterate over the allotted timeframe for trajectory generation
            with torch.no_grad():                                                           # Temporarily suspend gradient tracking globally to vastly accelerate collection
                batch = Batch.from_data_list(obs).to(device)                                # Convert the list of observations (one per environment) into a single batched graph representation suitable for input to the GNN agent.
                masks = env.get_masks()                                                     # shape (num_envs, max_atoms) - bool action masks for each environment (which actions are valid for each environment).
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")): # Engage the hardware accelerated mixed precision context
                    # Evaluate states to retrieve masked logits and value baselines for each environment. The agent outputs logits for the action distribution and value estimates for the critic.
                    logits, values = agent(batch.x, batch.edge_index, batch.edge_attr, batch.batch, masks) 

                # Convert the action logits for each environment into a categorical distribution (softmax) 
                # and sample actions for each environment (used to step the environment forward).
                dist = Categorical(logits=logits)                                           # Create a categorical distribution over actions for each environment based on the logits output by the agent.
                actions = dist.sample()                                                     # Draw specific discrete commands by sampling the calculated probability distribution
                log_probs = dist.log_prob(actions)                                          # Extract the numerical logarithmic probabilities of the actually sampled actions

            # Step all the environments forward with the sampled actions
            next_obs, rewards, dones, infos = env.step(actions, curr_ratio)                 

            # Loop over each distinct simulated environment instance
            for e in range(CONFIG["num_envs"]):                                             
                # Pack the current state of the environment (graph representation and mask) for environment e to store in the PPO buffer.
                state_pack = (obs[e].x, obs[e].edge_index, obs[e].edge_attr, masks[e])      # Bind specific tuple containing PyG tensors
                # Push an entire discrete transitional timestep package deep into the tracking buffers
                buffer.push(                                                                
                    env_id=e,                                                               # Pass the integer target environment channel
                    state=state_pack,                                                       # Store the input graph package
                    action=int(actions[e].item()),                                          # Store the raw scalar choice
                    reward=float(rewards[e].item()),                                        # Store the floating point heuristic return
                    done=float(dones[e].item()),                                            # Store the binary float termination flag
                    log_prob=float(log_probs[e].item()),                                    # Store the scalar log probability
                    value=float(values[e].item()),                                          # Store the critic's baseline value estimation
                )                                                                           

                # If the episode for environment e has ended (done), extract the terminal SMILES from the info dictionary,
                # add it to the top-k list along with its reward, and store the reward for this episode (policy quality).
                if dones[e].item() > 0.5:                                                   
                    term_smi = infos[e].get("terminal_smiles", "INVALID")                   
                    topk.add(float(rewards[e].item()), term_smi)                            
                    ep_term_rewards.append(float(rewards[e].item()))        
            # Overwrite historical observational tensors with the newly advanced batch
            obs = next_obs                                                                  

        # ---------------------------------------------------------------------------------
        # Bootstrapping & Advantage Estimation
        # Guess values for non-terminated edges and compute the GAE advantages.
        # ---------------------------------------------------------------------------------
        with torch.no_grad():                                                               # Suspend PyTorch's autograd tracker again for simple numerical inference
            batch = Batch.from_data_list(obs).to(device)                                    # Transform the final dangling observations into a unified graph batch tensor
            masks = env.get_masks()                                                         # Pull the final validity boolean masking tensor from the RDKit simulator
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")): # Engage the high speed mixed precision tensor operations mode
                _, last_values = agent(batch.x, batch.edge_index, batch.edge_attr, batch.batch, masks) # bootstrap the value of the final state for any episodes that haven't ended yet.

        # Compute advantages and returns using GAE for each environment's trajectory 
        # and flatten the collected transitions into tensors for training.
        buffer.finalize(last_values)                                                        

        # ---------------------------------------------------------------------------------
        # Proximal Policy Optimization
        # Repeatedly update weights using clipped surrogates and entropy bonuses.
        # ---------------------------------------------------------------------------------
        approx_kls, clip_fracs, losses = [], [], []                                         # Initialize empty tracking arrays for deep network diagnostics
        ent_coef = entropy_coef(update)                                                     # Compute the exact entropy scalar multiplier for this specific epoch

        # Extract the raw computed Generalized Advantage Estimation tensor, and normalize it.
        adv = buffer.advantages                                                             
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)                                       # Normalize advantages to have mean 0 and std 1 for better training stability.

        # For each inner optimization epoch, iterate over the entire buffer in randomized 
        # mini-batches to update the policy and value networks.
        for _epoch in range(CONFIG["ppo_epochs"]):                                          # Iterate inner optimization loops repeatedly processing the same locked dataset
            epoch_kls = []                                                                  # Initialize a temporary array to track KL divergence specifically for this inner iteration
            # Invoke buffer generator to slice the full dataset into randomized (if shuffle=True) sub-batches.
            # Each sub-batch contains a list of indices that correspond to the transitions stored in the buffer. 
            # Random mix of molecules, timesteps, and environments, to break the temporal correlation of the trajectories (standard requirement for stable PPO training).
            for idx in buffer.get_batches(CONFIG["batch_size"], shuffle=True):              
                idx_list = idx.tolist()                                                     # Convert torch index tensor into standard python list for arbitrary array accesses

                # Creates a list of Data objects for the current batch of indices, where each Data object 
                # contains the node features, edge index, and edge attributes for a single environment's state. 
                batch_list = [                                                              
                    Data(x=buffer.states[i][0], edge_index=buffer.states[i][1], edge_attr=buffer.states[i][2]) # Re-package stored PyG tuples into native PyG Data structures
                    for i in idx_list                                                       # Iterate exclusively over the specific index list
                ]                                                                           
                
                batch_masks = torch.stack([buffer.states[i][3] for i in idx_list], dim=0)   # Stack the action masks for the current batch of indices into a single tensor of shape (batch_size, max_atoms).
                batch_graph = Batch.from_data_list(batch_list).to(device)                   # This list is converted into a Batch object that can be processed by the GNN agent in a single forward pass. 

                # Extract the relevant tensors for the current mini-batch from the buffer, 
                # including actions, old log probabilities, returns, and advantages.
                actions_b = buffer.actions[idx]                                             # Extract the historic action tensors associated with this specific mini-batch
                old_logp_b = buffer.log_probs[idx]                                          # Extract the historically calculated logarithmic probabilities of those actions
                returns_b = buffer.returns[idx]                                             # Extract the absolute true returns calculated recursively over the episode
                adv_b = adv[idx]                                                            # Extract the newly zero-mean normalized advantage estimations

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")): # Open the mixed precision context window targeting GPU execution
                    logits, values = agent(                                                 # Feed forward pass routing batch graph vectors into the neural network
                        batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch, batch_masks # Input features arrays
                    )                                                                       # Close inference call
                    dist = Categorical(logits=logits)                                       # Restructure the updated raw logits into proper probability distributions

                    logp = dist.log_prob(actions_b)                                         # Calculate the log probabilities of the actions taken in the batch according to the current policy.
                    ratio = torch.exp(logp - old_logp_b)                                    # Calculate the probability ratio for PPO (=exponent of the difference between the new log probabilities and the old log probabilities).
                    
                    # Policy loss is computed as the negative of the minimum between the 
                    # unbounded surrogate and the clipped surrogate, averaged over the batch.
                    surr1 = ratio * adv_b                                                   # Compute the unbounded primary surrogate policy objective directly
                    surr2 = torch.clamp(ratio, 1.0 - CONFIG["clip_epsilon"], 1.0 + CONFIG["clip_epsilon"]) * adv_b # Compute the severely restricted clipped surrogate objective
                    policy_loss = -torch.min(surr1, surr2).mean()                           # PPO policy loss with clipping: select pessimistic bound and invert for descent

                    # Value loss is computed as the maximum between the squared error of the current value 
                    # estimates and the squared error of the clipped value estimates, averaged over the batch.
                    values_old = buffer.values[idx]                                         # Retrieve the historic value expectations stored in the replay memory
                    values_clipped = values_old + torch.clamp(                              # Compute pessimistic boundary constraining how fast the critic is allowed to update
                        values - values_old, -CONFIG["clip_epsilon"], CONFIG["clip_epsilon"]# Clamp delta to hyperparameter threshold
                    )                                                                       # Close clipped value boundary
                    vloss1 = (values - returns_b).pow(2)                                    # Calculate raw Mean Squared Error against true returns
                    vloss2 = (values_clipped - returns_b).pow(2)                            # Calculate heavily penalized MSE using the restricted clipped estimations
                    value_loss = 0.5 * torch.max(vloss1, vloss2).mean()                     # PPO value loss with clipping: choose the worst case bounding error to optimize

                    # Entropy is the expectation of the negative log probability of the current policy, averaged over the batch. 
                    entropy = dist.entropy().mean()                                             # Extract the numeric average entropy of the newly parameterized distribution
                    
                    # Loss is the weighted sum of the policy loss, value loss, and entropy, where the value loss 
                    # is scaled by a coefficient and the entropy is subtracted to encourage exploration in action selection.
                    loss = policy_loss + CONFIG["value_coef"] * value_loss - ent_coef * entropy # Aggregate actor error, scaled critic error, and subtracted exploration bonus

                # Backpropagate the loss through the network and update the network weights.
                optimizer.zero_grad(set_to_none=True)                                       # Erase historical accumulated gradient tensors to prevent compounding (set_to_none is faster and frees memory)
                scaler.scale(loss).backward()                                               # Execute AMP-scaled backpropagation populating the network tree with gradients
                scaler.unscale_(optimizer)                                                  # Unscale gradients before clipping to ensure that the clipping is done in the correct scale.
                torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG["max_grad_norm"]) # Heavily clip absolute magnitude of gradient vectors to arrest instability
                scaler.step(optimizer)                                                      # Inject the processed gradient deltas back into network weights
                scaler.update()                                                             # Command the AMP engine to evaluate internal multiplier dynamics

                # Since PPO re-uses the same batch of data for multiple epochs (ppo_epochs). If we update the network
                # too many times on the same data, the policy will drift too far from the old policy (Trust Region breakdown).
                with torch.no_grad():                                                       # Bypass autograd engine to quickly calculate secondary tracking metrics
                    # Compute the approximate KL divergence between the old and new policy for the current batch,
                    # used for monitoring and early stopping of PPO updates if the policy changes too much for this batch.
                    kl = (old_logp_b - logp).mean().item()                                  # Approximate KL divergence between the old and new policy for the current batch, used for monitoring and early stopping of PPO updates if the policy changes too much.
                    epoch_kls.append(kl)                                                    # Push KL scalar to the epoch averaging accumulator
                    # Calculate and save the fraction of actions in the batch for which the probability ratio was outside
                    # the clipping range, used as a diagnostic metric to understand how much the policy is changing during updates.
                    clip_frac = (torch.abs(ratio - 1.0) > CONFIG["clip_epsilon"]).float().mean().item() # Fraction of actions in the batch for which the probability ratio was outside the clipping range, used as a diagnostic metric to understand how much the policy is changing during updates.
                    clip_fracs.append(clip_frac)                                            # Push clip fraction to the diagnostic averaging array
                    losses.append(loss.item())                                              # Push the absolute scalar composite loss value to the tracker

            # Early Stopping: If the average KL divergence for this epoch exceeds the target KL specified 
            # in the CONFIG, break out of the PPO update loop (guard against catastrophic unlearning).
            if CONFIG["target_kl"] is not None:                                                 # If the target KL divergence is specified in the CONFIG
                mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0                       # Mathematically average out the KL drift spanning the inner batch loops
                approx_kls.append(mean_kl)                                                      # Add computed average to the master loop logging arrays
                if mean_kl > CONFIG["target_kl"]:                                               # If the average KL divergence for this epoch exceeds the target KL specified in the CONFIG, break out of the PPO update loop.
                    break                                                                       # Violently terminate internal learning epochs guarding against catastrophic unlearning

        # ---------------------------------------------------------------------------------
        # Logging & Model Evaluation
        # Compute real-time analytics, emit CLI updates, and log via WandB.
        # ---------------------------------------------------------------------------------
        if update % CONFIG["log_every"] == 0:                                               # Engage the logging logic branch strictly on defined intervals
            # Get the current batch of generated molecules (SMILES) from the environment 
            # for computation of validity, uniqueness, and diversity metrics.
            smiles_snapshot = env.get_smiles()                                              
            valid_r, uniq_r, div = tanimoto_diversity(smiles_snapshot)                      # Calculate validity, uniqueness, and diversity metrics for the current batch of generated molecules.

            best_r, best_s = topk.best()                                                    # Retrieve the historical high-water mark for strictly informational logging

            # Explained variance (EV) of the value function predictions: how well the critic network
            # is fitting the returns, or equivalently on predicting the future reward, with max=1.0.
            with torch.no_grad():                                                           
                # "Ground truth": the actual discounted cumulative rewards the agent ended up collecting during the episode.
                y_true = buffer.returns                                                     # Bind true cumulative returns vector for EV check
                # "Predictions": the value estimates produced by the critic network for each state in the buffer.
                y_pred = buffer.values                                                      # Bind base network predictions vector for EV check
                # Compute the explained variance (EV): which expresses how much of the variance in the true returns is captured by the predicted values.
                ev = 1.0 - torch.var(y_true - y_pred) / (torch.var(y_true) + 1e-8)          # Mathematically solve the standard Explained Variance fraction
                ev = float(ev.item())                                                       # Extract safely to python primitive

            # Stage-2 Reward Diagnostics
            # Extract the info dictionaries (P, A, D, aff_hat_z, aff_unc_z) from each sub-environment (empty if the episode has not terminated yet).
            _infos = [i for i in env.last_reward_infos() if i]                                      # Extract the cache of info dictionaries spanning the most recent episodic terminations
            # Local helper function that safely computes means for a specific dictionary key (could be any from "P" to "aff_unc_z"). 
            # Fallback safely to a mathematical NaN if the key is not present.
            def _meankey(k):                                                                        # Define a local helper function to safely compute means for specific reward components
                vals = [i[k] for i in _infos if i.get(k) is not None]                               # Extract valid numerical values for the requested tracking key from all info dicts
                return float(np.mean(vals)) if vals else float("nan")                               # Compute the arithmetic mean or fallback safely to a mathematical NaN
            # Compute the mean values for the "A", "D", "aff_hat_z", and "aff_unc_z" keys 
            # from the most recent reward diagnostics from the info dictionaries.
            s2_aff_hat = _meankey("aff_hat_z"); s2_aff_unc = _meankey("aff_unc_z")                  # Compute means for the z-scored affinity predictions and surrogate ensemble uncertainties
            s2_aff_term = _meankey("A"); s2_div_pen = _meankey("D")                                 # Compute means for the final weighted affinity term and the diversity penalty term
            
            wandb.log(                                                                      # Package data and emit remote network POST over to Weights & Biases
                {                                                                           # Begin payload dict
                    "update": update,                                                       # Log master iteration loop
                    "curriculum_ratio": curr_ratio,                                         # Log fractional progression through reward curriculum
                    "entropy_coef": ent_coef,                                               # Log dynamic exploration bonus scalar
                    "reward/step_mean": float(buffer.rewards.mean().item()),                # Mean reward over ALL buffer transitions (dominated by small per-step rewards; keep, but do not headline)
                    "reward/terminal_mean": (float(np.mean(ep_term_rewards)) if ep_term_rewards else float("nan")),         # Mean terminal reward across episodes that finished this update (true policy quality)
                    "reward/terminal_median": (float(np.median(ep_term_rewards)) if ep_term_rewards else float("nan")),     # Median terminal reward across episodes finished this update (robust to lucky jackpots)
                    "gen/sa_mean": _snapshot_sa(smiles_snapshot),                           # Log the SA score of what the policy emits now
                    "gen/n_terminated": len(ep_term_rewards),                               # Number of terminated episodes
                    "reward/std": float(buffer.rewards.std().item()),                       # Log standard deviation of the collected rewards
                    "eval/valid_rate": valid_r,                                             # Log fraction of string graphs that compiled successfully
                    "eval/unique_rate": uniq_r,                                             # Log fraction of the valid subset that were non-repeating
                    "eval/diversity": div,                                                  # Log subset similarity utilizing morgan fingerprints
                    "ppo/kl": float(np.mean(approx_kls)) if approx_kls else 0.0,            # Log policy drift metric
                    "ppo/clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,     # Log severity of the clipping intervention
                    "ppo/loss": float(np.mean(losses)) if losses else 0.0,                  # Log combined neural cost function
                    "critic/explained_var": ev,                                             # Log algorithmic prediction accuracy
                    "best/reward": best_r,                                                  # Log absolute theoretical best find
                    "affinity/pred_z_mean": s2_aff_hat,                                     # Stage-2: mean predicted z-scored pChEMBL of terminated molecules (NaN if affinity off)
                    "affinity/uncertainty_mean": s2_aff_unc,                                # Stage-2: mean ensemble uncertainty (epistemic) of terminated molecules
                    "affinity/term_mean": s2_aff_term,                                      # Stage-2: mean affinity reward term A after the uncertainty penalty
                    "diversity/penalty_mean": s2_div_pen,                                   # Stage-2: mean Tanimoto-to-archive penalty D (0 novel .. 1 duplicate)
                }                                                                           
            )                                                                               # Terminate WandB POST operation
            
            # Duplicate all metrics to the physical local textual file logger
            logger.log_step(                                                                
                update=update,                                                              # Inject step id
                r_mean=float(buffer.rewards.mean().item()),                                 # Inject return mean
                r_std=float(buffer.rewards.std().item()),                                   # Inject return deviation
                valid_rate=valid_r,                                                         # Inject topology validation
                unique_rate=uniq_r,                                                         # Inject non-collapsed string set fraction
                diversity=div,                                                              # Inject similarity proxy
                kl=float(np.mean(approx_kls)) if approx_kls else 0.0,                       # Inject drift score
                clip_frac=float(np.mean(clip_fracs)) if clip_fracs else 0.0,                # Inject guardrail triggers
                exp_var=ev,                                                                 # Inject EV score
                entropy_coef_val=ent_coef,                                                  # Inject scheduled exploration
                curriculum_ratio=curr_ratio,                                                # Inject objective mix
                best_reward=best_r,                                                         # Inject historic elite baseline
                best_smiles=best_s,                                                         # Inject absolute best sequence string
            )                                                                               

            # Print a concise summary of the current training state to the console for quick monitoring.
            print(                                                                          
                f"[{update:04d}] "                                                          # Print padded epoch format
                f"R={buffer.rewards.mean().item():.3f} "                                    # Print reward average
                f"(std={buffer.rewards.std().item():.3f})  "                                # Print reward deviation
                f"valid={valid_r:.3f}  uniq={uniq_r:.3f}  div={div:.3f}  "                  # Print standard metric trio
                f"kl={(float(np.mean(approx_kls)) if approx_kls else 0.0):.4f}  "           # Print deviation kl
                f"clip={(float(np.mean(clip_fracs)) if clip_fracs else 0.0):.3f}  "         # Print clip bounds limit
                f"ev={ev:.3f}  ent={ent_coef:.3f}  curr={curr_ratio:.3f}  "                 # Print algorithmic diagnostics
                f"best={best_r:.3f}  {best_s}"                                              # Print best candidate summary
            )                                                                               

        # Save a checkpoint of the current model, optimizer, scaler state, and top-k molecules every save_every updates.
        if update % CONFIG["save_every"] == 0 and update > 0:                               
            save_topk_txt(os.path.join(out_dir, f"topk_update_{update}.txt"), topk.to_rows()) # Export a human-readable ASCII table snapshot of the elite tracker
            ckpt = {                                                                        # Formulate a deep python dictionary housing complete network state
                "config": {**CONFIG, "num_actions": num_actions, "input_feats": input_feats}, # Bundle configuration hyper-parameters alongside active shapes
                "model": agent.state_dict(),                                                # Clone all trainable floating point vectors inside GNN and heads
                "optimizer": optimizer.state_dict(),                                        # Export momentum matrices tracking adam internal acceleration states
                "scaler": scaler.state_dict() if scaler is not None else None,              # Export AMP scaling window
                "update": update,                                                           # Stamp exact chronological marker
            }                                                                               # Close dictionary
            torch.save(ckpt, os.path.join(out_dir, f"checkpoint_{update}.pt"))              # Serialize the python dictionary and physically write to solid state memory

    # -------------------------------------------------------------------------------------
    # Training Teardown
    # Ensure artifacts are dumped securely upon standard loop exit.
    # -------------------------------------------------------------------------------------
    save_topk_txt(os.path.join(out_dir, "topk_final.txt"), topk.to_rows())                  # Persist final, ultimate edition of elite molecules list cleanly
    wandb.finish()                                                                          # Safely disconnect from telemetry servers flushing pending network ops


if __name__ == "__main__":                                                                  # Shield internal modules from unintended execution during external imports
    train()                                                                                 # Formally invoke main procedure initializing full run