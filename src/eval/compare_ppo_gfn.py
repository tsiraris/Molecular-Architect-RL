"""
=====================================================================
PPO vs GFlowNet Diversity and Reward Comparison (Stage-3 Evaluation)
=====================================================================

This script evaluates and compares two distinct generative policies—Proximal Policy Optimization (PPO) 
and Generative Flow Networks (GFlowNet)—to assess their respective exploration/exploitation trade-offs. 
It requires two input CSV files (one per method), each containing SMILES strings and their evaluated 
composite rewards scored by the exact same reward function.

By analyzing these populations, the script computes key metrics to render a diversity/reward Pareto view:
  1. Single best reward: Measures peak exploitation capability.
  2. Number of distinct Bemis-Murcko scaffolds above a reward threshold: Measures mode coverage and exploration.
  3. Top-k internal diversity: Measures local structural variety using mean pairwise (1 - Tanimoto similarity).
  4. Number of unique molecules above threshold: Measures the sheer volume of viable, distinct candidates.

Expected outcome: GFlowNet generally equals or outperforms PPO on mode coverage and top-k diversity at comparable
top rewards, while PPO often slightly edges out the single-best reward due to aggressive exploitation. The script
ensures honest, empirical reporting of these characteristics by exporting the results to a structured JSON file.
"""

import argparse
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold(smi):
    """
    Extracts the Bemis-Murcko scaffold from a given SMILES string.
    
    Translates the SMILES string into an RDKit molecule object. If the parsing is successful, 
    it delegates to RDKit's MurckoScaffold module to strip away side chains and return the 
    SMILES string of the isolated core ring-and-linker scaffold. This is used to define 
    distinct chemical "modes" for coverage metrics.
    
    Args:
        smi (str): The input SMILES string to process.
        
    Returns:
        str or None: The SMILES string of the molecule's scaffold, or None if the input is chemically invalid.
        
    Example:
        >>> _scaffold("CC1=CC=CC=C1")
        'c1ccccc1'
        >>> _scaffold("INVALID_SMILES")
        None
    """
    # -------------------------------------------------------------------------------------
    # Scaffold Extraction
    # Convert string to molecule and extract the structural core.
    # -------------------------------------------------------------------------------------
    m = Chem.MolFromSmiles(smi)                                                             # Parse the raw SMILES string into an RDKit molecule object
    # If the molecule is valid, compute and return its Bemis-Murcko scaffold SMILES; otherwise, return None
    # Reminder: Bemis-Murcko scaffolds are defined as the union of ring systems and linkers, excluding side chains.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None                        # Compute and return the scaffold SMILES if the molecule is valid, otherwise return None


def _int_div(smiles):
    """
    Calculates the internal topological diversity (1 - mean pairwise Tanimoto similarity) 
    of a population/batch of molecules.

    Converts a list of SMILES strings into RDKit molecules, drops invalid ones, and computes 
    Morgan fingerprints (radius 2, 2048 bits). It then computes the Tanimoto similarity 
    between all unique pairs in the set. The final internal diversity is defined as 
    1.0 minus the mean of these pairwise similarities.
    
    Args:
        smiles (List[str]): A list of SMILES strings representing the molecular batch.
        
    Returns:
        float: The mean pairwise internal diversity score (0.0 to 1.0). Returns 0.0 if fewer 
        than 2 valid molecules exist.
        
    Example:
        >>> pop = ["CCO", "CCN", "CCC"]
        >>> div = _int_div(pop)
        >>> round(div, 2)
        0.56
    """
    # -------------------------------------------------------------------------------------
    # Fingerprint Generation
    # Parse SMILES, filter invalids, and compute bit-vector representations.
    # -------------------------------------------------------------------------------------
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)            # Generate a 2048-bit Morgan fingerprint with radius 2 for each valid molecule
           for s in smiles if Chem.MolFromSmiles(s)]                                        # Filter out any chemically invalid SMILES strings during the list comprehension
    if len(fps) < 2:                                                                        # Check if the list contains enough valid fingerprints to perform pairwise comparisons
        return 0.0                                                                          # Return zero diversity if a meaningful pairwise comparison is impossible

    # -------------------------------------------------------------------------------------
    # Pairwise Similarity Calculation
    # Compute the Tanimoto similarity for all unique pairs to determine population variance.
    # -------------------------------------------------------------------------------------
    sims = []                                                                               # Initialize an empty list to accumulate all individual pairwise similarity scores
    for i in range(1, len(fps)):                                                            # Iterate through the fingerprints starting from the second element
        sims += list(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))                   # Compute similarity between the current fingerprint and all preceding ones, extending the list
    return float(1 - np.mean(sims))                                                         # Calculate the mathematical mean of similarities, subtract from 1 for diversity, and cast to float


def summarise(df, thresh, topk=50):
    """
    Computes summary evaluation metrics for a single generative model's (PPO or GFlowNet) output dataframe.
    
    Drops any rows missing a reward score and sorts the dataframe in descending order of reward. 
    It filters for molecules exceeding the `thresh` to calculate mode coverage (unique scaffolds) 
    and unique viable molecules. It isolates the top `topk` molecules to calculate peak 
    exploitation (single best reward) and local search diversity (top-k internal diversity).
    
    Args:
        df (pandas.DataFrame): A DataFrame containing at minimum 'smiles' and 'reward' columns.
        thresh (float): The minimum reward threshold a molecule must meet to be considered "successful".
        topk (int, optional): The number of elite molecules to analyze for local diversity. Defaults to 50.
        
    Returns:
        dict: A dictionary mapping evaluation metric names to their computed scalar values.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"smiles": ["C", "CC", "CCC"], "reward": [5.0, 7.0, 8.0]})
        >>> metrics = summarise(df, thresh=6.0, topk=2)
        >>> metrics["single_best_reward"]
        8.0
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Data Filtering and Sorting: Drop null reward rows, and create a subset (hi) with the reward >= "thresh" rows,
    # extract the unique Bemis-Murcko scaffolds from the hi subset and keep the top `topk` rows on another subset (top). 
    # ------------------------------------------------------------------------------------------------------------------
    df = df.dropna(subset=["reward"]).sort_values("reward", ascending=False)                # Remove rows with null rewards and sort the remainder from highest reward to lowest
    hi = df[df.reward >= thresh]                                                            # Create a subset dataframe containing strictly molecules that meet or exceed the success threshold
    scaffolds = {s for s in (_scaffold(x) for x in hi.smiles) if s}                         # Extract and accumulate a mathematical set of unique valid scaffolds from the high-performing subset
    top = df.head(topk)                                                                     # Slice the absolute best `topk` rows to form the elite subset for peak metric evaluation
    
    # ----------------------------------------------------------------------------------------------
    # Metric Aggregation: Compute the final four evaluation scalars and return them in a dictionary. 
    # The metrics are: - single best reward: the maximum reward found in the entire dataframe, 
    # - n_modes_above_thresh: number of unique Bemis-Murcko scaffolds above the threshold, 
    # - n_unique_above_thresh: number of structurally unique SMILES strings above the threshold, 
    # - topk_internal_diversity: internal Tanimoto diversity of the top `topk` molecules. 
    # ----------------------------------------------------------------------------------------------
    return {"single_best_reward": float(df.reward.max()) if len(df) else None,              # Extract the absolute peak reward found, or return None if the dataframe is entirely empty
            "n_modes_above_thresh": len(scaffolds),                                         # Count the total number of distinct Bemis-Murcko scaffolds discovered above the threshold
            "n_unique_above_thresh": int(hi.smiles.nunique()),                              # Count the absolute number of structurally unique SMILES strings above the threshold
            "topk_internal_diversity": round(_int_div(list(top.smiles)), 3)}                # Compute the internal Tanimoto diversity of the elite subset and round to 3 decimal places


def main():
    """
    The main execution wrapper providing a Command Line Interface (CLI) for the script.
    
    Parses CLI arguments specifying the input CSV files for PPO and GFlowNet, the reward 
    threshold, and the output JSON path. Loads the CSVs into pandas DataFrames, feeds them 
    through the `summarise` function, and writes the nested dictionary results into a JSON
    file and standard output.

    Args:
        None.
        
    Returns:
        None.
        
    Example:
        (From the command line)
        $ python compare_ppo_gfn.py --ppo_csv ppo.csv --gfn_csv gfn.csv --thresh 6.0
    """
    # -------------------------------------------------------------------------------------
    # Argument Parsing
    # Define and extract the required command-line inputs.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the Python standard library argument parser
    ap.add_argument("--ppo_csv", required=True, help="columns: smiles,reward")              # Register the required input flag for the PPO trajectory results CSV
    ap.add_argument("--gfn_csv", required=True, help="columns: smiles,reward")              # Register the required input flag for the GFlowNet trajectory results CSV
    ap.add_argument("--thresh", type=float, default=6.0)                                    # Register the optional threshold flag to define what constitutes a "success", defaulting to 6.0
    ap.add_argument("--out", default="../results/ppo_vs_gfn.json")                          # Register the optional output path flag indicating where to save the JSON report
    args = ap.parse_args()                                                                  # Execute the parser to extract the provided arguments into a namespace object
    
    # -------------------------------------------------------------------------------------
    # Data Processing and Output
    # Process both datasets (PPO and GFlowNet), merge the results, and export to disk.
    # -------------------------------------------------------------------------------------
    ppo = summarise(pd.read_csv(args.ppo_csv), args.thresh)                                 # Load the PPO CSV into memory and compute its summary metrics dictionary
    gfn = summarise(pd.read_csv(args.gfn_csv), args.thresh)                                 # Load the GFlowNet CSV into memory and compute its summary metrics dictionary
    res = {"threshold": args.thresh, "PPO": ppo, "GFlowNet": gfn}                           # Aggregate the exact threshold used alongside both sets of metrics into a master dictionary
    json.dump(res, open(args.out, "w"), indent=2)                                           # Open the designated output file and serialize the master dictionary as cleanly indented JSON
    print(json.dumps(res, indent=2))                                                        # Echo the nicely formatted JSON string to standard output for immediate terminal review


if __name__ == "__main__":                                                                  # Ensure the script is being run directly rather than imported as a module
    main()                                                                                  # Invoke the main CLI wrapper function