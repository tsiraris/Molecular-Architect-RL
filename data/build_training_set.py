"""
===================================
Surrogate Training Dataset Builder
===================================

This script merges two complementary affinity label sources into a single, unified 
surrogate training dataset (CSV) with a coherent "higher = better binder" target scale.

The RL agent's reward function relies on a fast surrogate model to estimate binding affinity. 
This script builds the training data for that surrogate by combining:
  1. Experimental data: ChEMBL pChEMBL values (higher = better).
  2. Computational data: gnina docking labels (lower kcal/mol = better, or CNN affinity).

Because these two scales differ fundamentally, each source is z-normalized independently 
so neither dominates the dataset. The physical docking scores (kcal/mol) are sign-flipped 
so that "higher = better" aligns with the pChEMBL logic. Rows are then concatenated. 
If a molecule is present in both sources, its z-scores are averaged. The resulting unified 
`smiles, pchembl` CSV provides a single binding-propensity z-score for the surrogate to learn.

Run Example:
    python ../data/build_training_set.py --chembl ../data/chembl_kras.csv \
        --docking ../data/docking_labels.csv --out ../data/surrogate_train.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem

# -----------------------------------------------------------------------------------------
# Data Normalization & Standardization Helpers
# Functions to ensure chemical strings and statistical distributions are uniform.
# -----------------------------------------------------------------------------------------

def _canon(s):
    """
    Converts a raw SMILES string into its canonicalized form using RDKit.
        
    Attempts to parse the input string into an RDKit Mol object. If successful, 
    it converts it back into a canonical SMILES string, ensuring that structurally 
    identical molecules have the exact same string representation for deduplication.
    
    Args:
        s (str): The raw SMILES string to be canonicalized.
        
    Returns:
        str or None: The canonical SMILES string, or None if the input is invalid.
        
    Example:
        >>> _canon("C1=CC=CC=C1")
        'c1ccccc1'
        >>> _canon("INVALID")
        None
    """
    m = Chem.MolFromSmiles(str(s))                                                          # Attempt to parse the casted string into a topological RDKit Mol object
    return Chem.MolToSmiles(m) if m is not None else None                                   # Return the canonical SMILES string if parsing succeeded, else return None


def _znorm(series):
    """
    Applies z-score normalization to a pandas Series (1-D labeled array).
    
    Extracts the numerical values, computes the mean and standard deviation, and 
    centers the data (subtracts mean, divides by standard deviation). A small epsilon 
    is added to the standard deviation to prevent zero-division errors.
    
    Args:
        series (pd.Series): A pandas Series containing numerical affinity labels.
        
    Returns:
        np.ndarray: A numpy array of the z-normalized values.
        
    Example:
        >>> s = pd.Series([1.0, 2.0, 3.0])
        >>> _znorm(s)
        array([-1.22474487,  0.        ,  1.22474487])
    """
    v = series.to_numpy(dtype=float)                                                        # Convert the input pandas Series into a standard float numpy array
    mu, sd = float(np.mean(v)), float(np.std(v) + 1e-8)                                     # Calculate the mean and standard deviation, adding epsilon to std for safety
    # Return the globally centered and scaled z-scores of the input Series
    return (v - mu) / sd                                                                    


def main():
    """
    Main execution pipeline for building the unified surrogate training set.
    
    Steps:
    1. Parses command-line arguments to locate the ChEMBL and Docking CSV files.
    2. Loads ChEMBL data, canonicalizes SMILES, applies z-normalization, and tags the source.
    3. Loads Docking data, selects either Vina scores (flips them) or CNN affinity, canonicalizes, z-normalizes, and tags.
    4. Concatenates both datasets into a single DataFrame.
    5. Deduplicates by averaging the z-scores of molecules appearing in both datasets.
    6. Saves the merged dataset to the specified output path.
    
    Args:
        None. (Reads from sys.argv via argparse).
        
    Returns:
        None. (Writes a CSV file to disk).
    """
    # -------------------------------------------------------------------------------------
    # Argument Parsing
    # Configure the CLI to accept paths to the independent data sources.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the command-line argument parser object
    ap.add_argument("--chembl", default=None, help="CSV with smiles,pchembl (higher=better)") # Define the path argument for the experimental ChEMBL dataset
    ap.add_argument("--docking", default=None, help="CSV with smiles,dock_affinity (lower=better)") # Define the path argument for the computational docking dataset
    ap.add_argument("--out", default="../data/surrogate_train.csv")                         # Define the destination path for the merged output CSV file
    ap.add_argument("--use_cnn", action="store_true",                                       # Define a boolean flag to toggle between physics-based docking and CNN scoring
                    help="if set, use gnina cnn_affinity (higher=better) instead of dock_affinity") # Provide help text explaining the toggle behavior
    args = ap.parse_args()                                                                  # Parse the provided command-line arguments into the args namespace

    frames = []                                                                             # Initialize an empty list to accumulate the processed DataFrames
    
    # ----------------------------------------------------------------------------------------------------
    # ChEMBL Data Processing: Load experimental data, canonicalize SMILES, and independently Z-normalize 
    # the pChEMBL values into 'label' column, creating a 3-column DataFrame ("smiles", "label", "source").
    # ----------------------------------------------------------------------------------------------------
    if args.chembl and os.path.exists(args.chembl):                                         # Check if the ChEMBL argument was provided and the file physically exists
        d = pd.read_csv(args.chembl)                                                        # Load the experimental dataset into a pandas DataFrame
        col = "pchembl" if "pchembl" in d.columns else d.columns[1]                         # Identify the label column (either strictly 'pchembl' or fallback to the 2nd column)
        d = d[["smiles", col]].dropna()                                                     # Subset the DataFrame to just the SMILES and label, dropping any missing rows
        # Canonicalize the SMILES (uniform representation) and drop any rows that failed to parse
        d["smiles"] = d["smiles"].map(_canon); d = d.dropna()                               # Apply canonicalization to all SMILES and drop any rows that failed to parse
        # Z-normalize the pChEMBL values into a new 'label' column and tag the data source ("chembl")
        d["label"] = _znorm(d[col]); d["source"] = "chembl"                                 
        frames.append(d[["smiles", "label", "source"]])                                     # Append the standardized 3-column DataFrame to the accumulation list
        print(f"[merge] chembl: {len(d)} rows")                                             # Log the total number of successfully processed ChEMBL rows to the console

    # ---------------------------------------------------------------------------------------------------------------
    # Docking Data Processing: Load computational docking data, canonicalize SMILES, handle "cnn_affinity" or 
    # "dock_affinity" selection, apply sign-flipping if necessary, z-normalize them into 'label', and tag the 
    # source ("docking"). The final DataFrame is also 3 columns: ("smiles", "label", "source").
    # ---------------------------------------------------------------------------------------------------------------
    if args.docking and os.path.exists(args.docking):                                       # Check if the Docking argument was provided and the file physically exists
        d = pd.read_csv(args.docking)                                                       # Load the computational docking dataset into a pandas DataFrame
        # If the user requested CNN scores and "cnn_affinity" exists in the file,
        # subset to just those columns and extract the raw affinity values. 
        if args.use_cnn and "cnn_affinity" in d.columns:                                    # Check if the user requested CNN scores and they exist in the file
            d = d[["smiles", "cnn_affinity"]].dropna(); raw = d["cnn_affinity"]             # higher=better # Subset to CNN scores and extract raw labels (higher is natively better)
        # else if "dock_affinity" exists, subset to those columns and flip the sign 
        # so that higher is better.
        else:                                                                               # Fallback to standard Vina physics-based docking scores (kcal/mol)
            d = d[["smiles", "dock_affinity"]].dropna(); raw = -d["dock_affinity"]          # flip: higher=better # Subset to docking scores and invert the sign so higher represents stronger binding
        # Canonicalize the SMILES (uniform representation) and drop any rows that failed to parse
        d["smiles"] = d["smiles"].map(_canon); d = d.dropna(subset=["smiles"])              # Apply canonicalization to all SMILES and drop rows where structural parsing failed
        # Z-normalize the flipped/selected raw values into a new 'label' column and tag the data source ("docking")
        d["label"] = _znorm(raw.loc[d.index]); d["source"] = "docking"                      # Z-normalize the flipped/selected raw values, align indices, and tag the source
        frames.append(d[["smiles", "label", "source"]])                                     # Append the standardized 3-column computational DataFrame to the list
        print(f"[merge] docking: {len(d)} rows")                                            # Log the total number of successfully processed Docking rows to the console

    # -------------------------------------------------------------------------------------
    # Merging & Saving: Vertically concatenate the two datasets, resolve molecules present 
    # in both datasets by averaging their z-score, and export for surrogate training.
    # -------------------------------------------------------------------------------------
    if not frames:                                                                          # Check if the accumulation list remained empty (no files loaded)
        raise SystemExit("Provide --chembl and/or --docking with existing files.")          # Terminate the script aggressively with an error instructing the user to provide valid inputs

    # Concatenate all loaded DataFrames vertically into one unified dataset
    allrows = pd.concat(frames, ignore_index=True)                                          
    # For molecules present in both sources, average their labels (per-source z-scores)
    merged = allrows.groupby("smiles", as_index=False)["label"].mean()                              # Group by identical SMILES strings and average their standardized labels to resolve overlaps
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)                                    # Ensure the target directory for the output file exists, creating it if necessary
    # Rename the label column to match downstream expectations ("label" -> "pchembl") and 
    # write the unified dataset to CSV
    merged.rename(columns={"label": "pchembl"}).to_csv(args.out, index=False)                       # Rename the label column to match downstream expectations and write to CSV
    print(f"[merge] wrote {len(merged)} unique molecules -> {args.out} (z-scored unified label)")   # Log the final unified dataset size and output path to the console


if __name__ == "__main__":
    main()