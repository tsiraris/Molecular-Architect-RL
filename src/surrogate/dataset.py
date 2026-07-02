"""
=============================================
Surrogate Model Dataset Preparation Pipeline
=============================================

This script processes a ChEMBL affinity CSV to generate featurized PyTorch Geometric (PyG) 
graphs suitable for training a deep Graph Neural Network (GNN) affinity surrogate.

Input CSV requirements (produced by data/fetch_chembl_kras.py):
    - `smiles`: The string representation of the molecule.
    - `pchembl`: The binding affinity (-log10(IC50/Ki/Kd in M)); higher = stronger binder.

Core features:
1. Leakage-Safe Bemis-Murcko Scaffold Split: Unlike random splitting (which allows near-duplicate 
   analogues to leak from the training set into the test set and artificially inflate scores), 
   splitting by core Murcko scaffold forces the surrogate to be evaluated on strictly novel 
   chemotypes. 
2. Label Standardization: Regression labels (`pchembl`) are strictly z-normalized (mean 0, std 1) 
   using only the statistics of the training set to prevent data leakage. These (mean, std) 
   statistics are preserved and saved alongside the trained model so that the RL reward loop 
   and subsequent evaluations can consistently invert and interpret the surrogate's predictions.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .featurize import to_data

# -----------------------------------------------------------------------------------------
# Scaffold Extraction
# Helper functions to identify the foundational ring systems of chemical structures.
# -----------------------------------------------------------------------------------------
def _murcko(smi: str) -> str:
    """
    Computes the canonical Bemis-Murcko scaffold SMILES for a given molecule.
    
    Parses the input SMILES string into an RDKit Mol object and extracts its underlying 
    framework (removing all side chains while retaining ring systems and linking bonds). 
    If the molecule is invalid or acyclic, it returns an empty string to group them together.
    
    Args:
        smi (str): The input SMILES sequence to process.
        
    Returns:
        str: The SMILES string of the isolated Murcko scaffold, or '' upon failure.
        
    Example:
        >>> _murcko("CC1=CC=CC=C1")
        'c1ccccc1'
    """
    # -------------------------------------------------------------------------------------
    # Murcko Parsing Logic
    # -------------------------------------------------------------------------------------
    m = Chem.MolFromSmiles(smi)                                                             # Parse the raw input SMILES string into an active RDKit Mol topological graph
    if m is None:                                                                           # Check if the RDKit compilation failed (e.g., due to invalid chemical valency)
        return ""                                                                           # Return an empty string fallback to safely handle chemically malformed inputs
    try:                                                                                    # Wrap the scaffold extraction in a try-block to catch deeper structural exceptions
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)                                   # Compute and return the stripped-down Bemis-Murcko scaffold string directly
    except Exception:                                                                       # Catch any unpredictable algorithmic failures during the topological traversal
        return ""                                                                           # Return the empty string fallback to group unprocessable molecules together


# -----------------------------------------------------------------------------------------
# Data Cleaning & Normalization
# Functions to sanitize raw biological assay data and handle chemical duplicates.
# -----------------------------------------------------------------------------------------
def load_clean(csv_path: str) -> pd.DataFrame:
    """
    Reads a raw affinity CSV, validates SMILES, canonicalizes structures, and deduplicates.
    
    Loads the dataframe and searches flexibly for `smiles` and `pchembl` columns. It attempts 
    to parse every SMILES; failures are dropped. Valid molecules are re-serialized to 
    ensure canonicalization. Finally, exact chemical duplicates are collapsed by computing 
    the median of their pChEMBL values (standard QSAR hygiene).
    
    Args:
        csv_path (str): The file path to the raw input CSV dataset.
        
    Returns:
        pd.DataFrame: A cleaned dataframe containing strictly canonical "smiles" and "pchembl".
        
    Example:
        >>> df = load_clean("data/kras_raw.csv")
        >>> list(df.columns)
        ['smiles', 'pchembl']
    """
    # -------------------------------------------------------------------------------------
    # File Loading & Column Standardization
    # Read and isolate the data ("smiles", "pchembl" columns) droping any NaN values.
    # -------------------------------------------------------------------------------------
    df = pd.read_csv(csv_path)                                                              # Read the specified comma-separated values file into a pandas DataFrame object
    cols = {c.lower(): c for c in df.columns}                                               # Create a lowercase mapping of all column headers to gracefully handle arbitrary casing
    smi_col = cols.get("smiles");  y_col = cols.get("pchembl") or cols.get("pchembl_value") # Extract exact original column names for SMILES and activity metrics safely
    if smi_col is None or y_col is None:                                                    # Check if either of the critically required columns are missing from the parsed mapping
        raise ValueError(f"CSV must have 'smiles' and 'pchembl' columns; found {list(df.columns)}") # Abort execution with an explicit error detailing the discovered incompatible headers
    df = df[[smi_col, y_col]].rename(columns={smi_col: "smiles", y_col: "pchembl"})         # Isolate only the relevant columns and strictly rename them to standard internal identifiers
    df = df.dropna()                                                                        # Drop any rows containing NaN values in either the sequence or target affinity columns
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Chemical Validation (Valid chemistry from SMILES to RDKit) & Canonicalization (Valid serialization from RDKit graph to SMILES)
    # Parse every SMILES into an RDKit graph object and re-serialize to SMILES to ensure validity and canonicalization, keeping only valid molecules.
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    keep = []                                                                               # Initialize an empty accumulator list to store tuples of sanitized data
    for smi, y in zip(df["smiles"], df["pchembl"]):                                         # Iterate concurrently over the raw SMILES strings and their corresponding biological labels
        m = Chem.MolFromSmiles(str(smi))                                                    # Coerce sequence to string and attempt compilation into a functional RDKit graph object
        if m is None:                                                                       # Check if the molecular graph initialization failed due to invalid chemistry
            continue                                                                        # Skip this specific record and proceed to the next iteration safely
        try:                                                                                # Wrap serialization to catch potential edge-case failures in RDKit's string generation
            keep.append((Chem.MolToSmiles(m), float(y)))                                    # Re-serialize into canonical SMILES, cast label to float, and append the valid tuple
        except Exception:                                                                   # Catch unhandled chemical topological failures during the serialization phase
            continue                                                                        # Silently bypass the problematic molecule to maintain batch integrity
            
    # -------------------------------------------------------------------------------------
    # Deduplication & Aggregation: Sanitized tuples back into DataFrame, dropping NaNs, and
    # collapse exact-duplicate molecules keeping their median activity 
    # -------------------------------------------------------------------------------------
    out = pd.DataFrame(keep, columns=["smiles", "pchembl"]).dropna()                        # Convert the sanitized tuples back into a structured pandas DataFrame and drop straggler NaNs
    # Collapse exact-duplicate molecules to their median pChEMBL value (standard QSAR hygiene)
    out = out.groupby("smiles", as_index=False)["pchembl"].median()                         # Group identical canonical SMILES together and resolve conflicting affinities via the median
    return out.reset_index(drop=True)                                                       # Drop the old hierarchical index array and return the flattened, perfectly clean DataFrame


# -----------------------------------------------------------------------------------------
# Dataset Splitting
# Implement structurally-aware dataset partitioning to strictly prevent test-set leakage.
# -----------------------------------------------------------------------------------------
def scaffold_split(df: pd.DataFrame, frac_train=0.8, frac_val=0.1, seed=42
                   ) -> Tuple[List[int], List[int], List[int]]:
    """
    Partitions the dataset into Train/Val/Test subsets strictly by Bemis-Murcko scaffold.
    
    Calculates the Bemis-Murcko scaffold for every molecule and groups their integer indices.
    It sorts these groups in descending order of size (with deterministic random tie-breaking). 
    It then sequentially fills the train, validation, and test sets. This ensures that the 
    largest, most common chemotypes anchor the training phase, while the surrogate model is 
    tested on entirely unseen, rarer core scaffolds.
    
    Args:
        df (pd.DataFrame): The cleaned dataframe containing a "smiles" column.
        frac_train (float, optional): Fraction of data for training. Defaults to 0.8.
        frac_val (float, optional): Fraction of data for validation. Defaults to 0.1.
        seed (int, optional): Random seed for deterministic subsetting. Defaults to 42.
        
    Returns:
        Tuple[List[int], List[int], List[int]]: Three lists containing integer row indices 
        corresponding to the train, val, and test splits respectively.
        
    Example:
        >>> tr, va, te = scaffold_split(clean_df)
        >>> len(set(tr).intersection(set(te)))
        0
    """
    # -------------------------------------------------------------------------------------
    # Scaffold Grouping: Compute the Bemis-Murcko scaffold for every molecule and map their 
    # dataframe index to its corresponding foundational scaffold string (a.k.a. group dict).
    # -------------------------------------------------------------------------------------
    groups: Dict[str, List[int]] = defaultdict(list)                                        # Initialize a dictionary mapping scaffold strings to a list of matching dataframe indices
    for i, smi in enumerate(df["smiles"]):                                                  # Iterate sequentially through the dataset sequences alongside their absolute row index
        groups[_murcko(smi)].append(i)                                                      # Compute the scaffold and append the current row index into the corresponding structural group
        
    # ---------------------------------------------------------------------------------------
    # Group Sorting & Split Allocation: Order groups deterministically by size and distribute
    # them into train, val, and test, with the train dataset anchoring the largest scaffolds.
    # ---------------------------------------------------------------------------------------
    rng = np.random.RandomState(seed)                                                       # Initialize an isolated Numpy random number generator with the specified deterministic seed
    scaf_sets = list(groups.values())                                                       # Extract all the index lists from the mapping, discarding the scaffold string keys
    # Shuffle within a deterministic order of decreasing group size
    scaf_sets.sort(key=lambda s: (-len(s), rng.random()))                                   # Sort sets descending by size, using the PRNG to deterministically break ties for uniform groups
    n = len(df); n_tr, n_va = int(frac_train * n), int(frac_val * n)                        # Calculate the absolute total dataset size and the exact target integer capacities for the splits
    train, val, test = [], [], []                                                           # Initialize three empty lists designed to accumulate the split-assigned dataframe indices
    for s in scaf_sets:                                                                     # Sequentially iterate over the sorted groups of structurally identical indices
        if len(train) + len(s) <= n_tr:                                                     # Check if adding the current group keeps the training set under or at its required capacity
            train += s                                                                      # Bulk append all indices within the current structural group into the training accumulator
        elif len(val) + len(s) <= n_tr + n_va - len(train) + len(val):                      # Check if adding the group keeps the validation set under its relative capacity threshold
            val += s                                                                        # Bulk append all indices within the current structural group into the validation accumulator
        else:                                                                               # If neither train nor validation can accommodate the group size
            test += s                                                                       # Default to dumping the remaining structural groups strictly into the final test accumulator
            
    # Safety: if rounding left a split empty, peel from train
    if not val:  val = train[-max(1, n // 10):]; train = train[:-len(val)]                  # If validation is empty, forcefully peel up to 10% from the tail of the training set
    if not test: test = train[-max(1, n // 10):]; train = train[:-len(test)]                # If test is empty, forcefully peel up to 10% from the adjusted tail of the training set
    return train, val, test                                                                 # Yield the three fully populated and structurally segregated integer index lists


def normalise_labels(y: np.ndarray, stats: Dict = None) -> Tuple[np.ndarray, Dict]:
    """
    Standardizes regression labels to zero-mean and unit-variance.
    
    If `stats` is None (i.e., processing the train set), it computes the empirical mean 
    and standard deviation. It then applies the z-score transformation `(y - mean) / std`. 
    If `stats` are provided (i.e., processing val/test sets), it strictly uses the provided 
    parameters to prevent statistical leakage from the evaluation sets.
    
    Args:
        y (np.ndarray): The raw 1D array of numeric target labels.
        stats (Dict, optional): Dictionary containing "mean" and "std". Defaults to None.
        
    Returns:
        Tuple[np.ndarray, Dict]: The z-normalized array and the statistics dictionary used.
        
    Example:
        >>> y_train = np.array([5.0, 6.0, 7.0])
        >>> z_train, stats = normalise_labels(y_train)
        >>> print(stats["mean"])
        6.0
    """
    # -------------------------------------------------------------------------------------
    # Z-Score Standardization: Compute base statistics if 'stats' (mean/std) are absent, 
    # and apply the normalization scaling on the provided label array.
    # -------------------------------------------------------------------------------------
    if stats is None:                                                                       # Check if explicit normalization statistics have been provided by the caller
        mean, std = float(np.mean(y)), float(np.std(y) + 1e-8)                              # Compute empirical mean and standard deviation, injecting an epsilon to avert division by zero
        stats = {"mean": mean, "std": std}                                                  # Bundle the newly calculated mathematical constants into the tracking dictionary
    z = (y - stats["mean"]) / stats["std"]                                                  # Execute the vectorized z-score transformation strictly using the resolved statistics dictionary
    return z, stats                                                                         # Return the mathematically scaled label array alongside its corresponding parameter dictionary


# -----------------------------------------------------------------------------------------
# Master Dataset Orchestrator
# High-level function uniting cleaning, splitting, normalizing, and PyG generation.
# -----------------------------------------------------------------------------------------
def build_datasets(csv_path: str, frac_train=0.8, frac_val=0.1, seed=42):
    """
    Executes the comprehensive data pipeline from raw CSV to graph-featurized PyG lists.
    
    1. Triggers `load_clean` to prepare the base dataframe.
    2. Triggers `scaffold_split` to securely partition indices.
    3. Extracts all pChEMBL labels and calculates normalizations strictly on the train indices.
    4. Iterates through the subsets, converting each SMILES into a PyTorch Geometric 
       `Data` object via `to_data`, attaching the properly normalized floating point label.
       
    Args:
        csv_path (str): The file path to the raw dataset.
        frac_train (float, optional): Fraction allocated to training. Defaults to 0.8.
        frac_val (float, optional): Fraction allocated to validation. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        Tuple[List[Data], List[Data], List[Data], Dict, pd.DataFrame]: The train/val/test 
        PyG dataset lists, the computed normalization dictionary, and the cleaned dataframe.
        
    Example:
        >>> tr, va, te, stats, df = build_datasets("data/kras.csv")
        >>> type(tr[0])
        <class 'torch_geometric.data.data.Data'>
    """
    # -------------------------------------------------------------------------------------
    # Pipeline Assembly & Execution
    # String together cleaning, partitioning, and statistical derivation sequentially.
    # -------------------------------------------------------------------------------------
    df = load_clean(csv_path)                                                               # Trigger the data hygiene subroutine to load, format, and deduplicate the source records
    tr, va, te = scaffold_split(df, frac_train, frac_val, seed)                             # Segment the cleaned dataframe indices securely utilizing the structural Bemis-Murcko rules
    y_all = df["pchembl"].to_numpy(dtype=np.float64)                                        # Extract the entire biological affinity column directly into a high-precision Numpy flat array
    _, stats = normalise_labels(y_all[tr])                                                  # Run normalization subroutine strictly on the training subset to extract safe, leakage-free stats

    # -------------------------------------------------------------------------------------
    # Dataset Featurization: Convert SMILES into PyTorch Geometric 'Data' objects.
    # -------------------------------------------------------------------------------------
    def make(idxs):                                                                         # Define an inner helper function to transform raw integer indices into processed PyG datasets
        out = []                                                                            # Initialize an empty array designed to accumulate the successfully compiled graph objects
        for i in idxs:                                                                      # Iterate sequentially over every requested dataframe index provided in the argument array
            d = to_data(Chem.MolFromSmiles(df["smiles"][i]),                                # Convert the specific row's sequence into an RDKit Mol and pipe it into the feature extractor
                        y=float((df["pchembl"][i] - stats["mean"]) / stats["std"]))         # Manually compute and attach the z-score normalized floating point label matching this structure
            if d is not None:                                                               # Validate that the feature extraction pipeline yielded a complete, non-null Data object
                out.append(d)                                                               # Push the validated PyTorch Geometric dataset instance into the active subset accumulator
        return out                                                                          # Return the fully populated array containing all successfully featurized subset graphs

    return make(tr), make(va), make(te), stats, df                                          # Execute featurization across all subsets and return alongside stats and the cleaned dataframe


# -----------------------------------------------------------------------------------------
# Normalization Disk I/O
# Utilities to save and load the z-score constants required for external inference.
# -----------------------------------------------------------------------------------------
def save_norm(stats: Dict, path: str):
    """
    Serializes the normalization statistics dictionary to disk.
    
    Extracts the parent directory from the specified path, safely creates it if absent, 
    and dumps the dictionary as a human-readable JSON string.
    
    Args:
        stats (Dict): The dictionary containing "mean" and "std".
        path (str): The destination file path.
        
    Returns:
        None
        
    Example:
        >>> save_norm({"mean": 5.0, "std": 1.2}, "models/stats.json")
    """
    # -------------------------------------------------------------------------------------
    # JSON Serialization
    # -------------------------------------------------------------------------------------
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)                                # Safely resolve and generate the parent directory structure on disk, bypassing exist errors
    with open(path, "w") as f:                                                              # Open the explicitly requested target file pathway operating in standard write mode
        json.dump(stats, f)                                                                 # Transcribe the in-memory python dictionary into a strictly formatted JSON text representation


def load_norm(path: str) -> Dict:
    """
    Deserializes the normalization statistics dictionary from disk.
    
    Opens the target JSON file and parses its contents back into a Python dictionary. 
    Required to re-initialize models for RL inference.
    
    Args:
        path (str): The file path to read from.
        
    Returns:
        Dict: The parsed statistics dictionary.
        
    Example:
        >>> stats = load_norm("models/stats.json")
        >>> stats["std"]
        1.2
    """
    # -------------------------------------------------------------------------------------
    # JSON Deserialization
    # -------------------------------------------------------------------------------------
    with open(path) as f:                                                                   # Open the specified target file pathway operating in default read-only text mode
        return json.load(f)                                                                 # Parse the textual JSON payload back into a functional python dictionary object and return