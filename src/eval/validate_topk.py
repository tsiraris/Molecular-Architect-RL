"""
==============================================
Top-K Independent Physics Validation Pipeline
==============================================

This script represents the frontier validation stage (Stage-2) of the generative pipeline. 
It rigorously evaluates the generator's final top-k molecules using independent, 
physics-based tools that the RL agent has never explicitly seen during training. This 
proves that the generated molecules can survive strict external scrutiny and aren't 
just exploiting the reward function.

For the generator's final top-k archive, it executes the following sequential pipeline:
  1. Surrogate Predicted pChEMBL: Evaluates the molecules against the proxy model.
  2. Real Re-docking (gnina/smina): Computes minimizedAffinity, CNNaffinity, CNNscore, and saves best poses.
  3. PoseBusters: Validates the generated poses to determine the percentage that are physically valid.
  4. PhysDock Bridge (Optional): Runs expensive Boltz-2 + OpenMM simulations on the very top handful.

It produces a comprehensive `results/topk_validation.csv` file and a printed summary 
containing the docking-score distribution and the percentage of PoseBusters-valid structures.

Usage Example:
    python -m eval.validate_topk --config ../configs/kras_g12c.yaml \
        --topk ../src/artifacts/<run>/topk_final.txt \
        --surrogate ../artifacts/surrogate_kras [--physdock /path/to/PhysDock]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem


def _read_topk(path):
    """
    Parses a formatted top-k tracking text file to extract valid, unique, canonical SMILES strings.
    
    Opens the target text file and reads it line by line. It automatically skips empty lines, 
    comments (marked with `#`), and table headers (starting with "reward"). It replaces pipe `|` 
    separators with spaces to handle tabular formats, splits the line, and checks tokens against 
    RDKit's SMILES parser. Successfully parsed strings are converted to canonical SMILES and 
    deduplicated using a hash set.
    
    Args:
        path (str): The absolute or relative file path to the top-k text file.
        
    Returns:
        List[str]: A list of guaranteed unique, valid, canonical SMILES strings.
        
    Example:
        >>> # Assuming 'topk.txt' contains: "9.5 | c1ccccc1"
        >>> unique_smiles = _read_topk("topk.txt")
        >>> print(unique_smiles)
        ['c1ccccc1']
    """
    # -----------------------------------------------------------------------------------------
    # Top-K File Parsing
    # Open the top-k text file, bypass headers/comments, and extract raw valid SMILES tokens.
    # -----------------------------------------------------------------------------------------
    smis = []                                                                               # Initialize an empty python list to accumulate the raw valid SMILES tokens
    with open(path) as f:                                                                   # Open the specified file path in standard read mode using a context manager
        for line in f:                                                                      # Iterate sequentially through every single line contained in the text file
            line = line.strip()                                                             # Strip leading and trailing whitespace/newlines from the current line
            if not line or line.startswith("#") or line.lower().startswith("reward"):       # Check if the line is empty, a comment, or the tabular header row
                continue                                                                    # Skip processing this line and immediately move to the next iteration
            # Replace pipe separators with spaces and split the line into individual string tokens
            for tok in line.replace("|", " ").split():                                      
                # Parse the token with RDKit to verify it is chemically valid
                if Chem.MolFromSmiles(tok) is not None:                                     
                    smis.append(tok); break                                                 # Append the valid token to the list and break the inner loop to process the next line
    
    # -----------------------------------------------------------------------------------------
    # Canonicalization & Deduplication
    # Process the raw tokens to ensure only unique, standardized structures are returned.
    # -----------------------------------------------------------------------------------------
    seen, out = set(), []                                                                   # Initialize a hash set to track seen molecules and a list for the final output
    # Iterate over the collected raw SMILES strings, convert them to Mol objects, 
    # and back to strings, adding only the unique ones to the tracking set
    for s in smis:                                                                          
        c = Chem.MolToSmiles(Chem.MolFromSmiles(s))                                         # Convert the string to a Mol object and back to string to enforce RDKit canonicalization
        if c not in seen:                                                                   # Check if this exact canonical string has not been encountered yet
            seen.add(c); out.append(c)                                                      # Add the novel string to the tracking set and append it to the final output list
    return out                                                                              # Return the fully deduplicated, canonicalized list of SMILES


def main():
    """
    The main execution orchestrator for the independent top-k physics validation pipeline.
    
    Parses command-line arguments to resolve paths to the configuration, top-k file, 
    surrogate model, and external docking binaries. It loads the top-k molecules into a 
    Pandas DataFrame. It sequentially feeds these molecules through the fast surrogate proxy, 
    the rigorous computational oracle (gnina/smina), and PoseBusters for physical sanity 
    checks. If a PhysDock path is provided, it bridges the top candidates to expensive 
    MD/Boltz-2 simulations. Finally, it exports all computed metrics to a CSV.
    
    Args:
        None (relies on sys.argv via argparse).
        
    Returns:
        None.
        
    Example:
        >>> # Executed via CLI:
        >>> # python -m eval.validate_topk --config kras.yaml --topk topk.txt
    """
    # -----------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Define and parse required and optional command-line arguments for the validation run.
    # -----------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the argument parser object to process command-line inputs
    ap.add_argument("--config", required=True)                                              # Define the required path argument for the main YAML configuration file
    ap.add_argument("--topk", required=True)                                                # Define the required path argument pointing to the agent's generated top-k text file
    ap.add_argument("--surrogate", default=None)                                            # Define the optional path to the trained predictive proxy model checkpoint
    ap.add_argument("--physdock", default=None, help="PhysDock repo for Boltz-2/OpenMM on the top handful") # Define optional path to PhysDock repo for deep physics validation
    ap.add_argument("--physdock_env", default="physdock")                                   # Define the Conda environment name to invoke when running PhysDock subprocesses
    ap.add_argument("--n_physics", type=int, default=5)                                     # Define the integer threshold for how many top molecules undergo expensive MD/Boltz-2
    ap.add_argument("--out", default="../results/topk_validation.csv")                      # Define the target output path for the finalized CSV validation report
    args = ap.parse_args()                                                                  # Parse the supplied CLI arguments and bind them to the args namespace

    # -----------------------------------------------------------------------------------------
    # Configuration & Data Loading
    # Load the YAML config, extract unique SMILES, and initialize the tracking DataFrame.
    # -----------------------------------------------------------------------------------------
    cfg = yaml.safe_load(open(args.config))                                                 # Safely read and parse the YAML configuration dictionary into memory
    smis = _read_topk(args.topk)                                                            # Parse the target text file to extract a clean list of unique SMILES strings
    print(f"[validate] {len(smis)} unique top-k molecules")                                 # Print a console notification confirming the total number of unique molecules loaded
    df = pd.DataFrame({"smiles": smis})                                                     # Initialize a pandas DataFrame with the SMILES list as the foundational column

    # -----------------------------------------------------------------------------------------
    # Step 1 - Surrogate Predictions
    # Re-evaluate the molecules using the proxy model to log predicted affinity and uncertainty.
    # -----------------------------------------------------------------------------------------
    if args.surrogate:                                                                      # Check if a surrogate model path was provided via the CLI
        from surrogate.predict import AffinityScorer                                        # Dynamically import the AffinityScorer class to avoid loading PyTorch unnecessarily
        sc = AffinityScorer(args.surrogate, device="cpu")                                   # Instantiate the scoring model on the CPU using the provided checkpoint path
        # Loops through the SMILES strings and converts all of them into a List of RDKit molecules,
        # scores the entire list through the surrogate model, and appends the predicted affinity and uncertainty to the DataFrame
        mu, sd = sc.score_mols([Chem.MolFromSmiles(s) for s in smis])                       # Convert SMILES to Mol objects and predict the mean affinity and standard deviation
        df["pred_pchembl"] = [round(sc.to_pchembl(z), 3) for z in mu]                       # Convert raw Z-scores to pChEMBL values, round them, and append as a DataFrame column
        df["pred_unc_z"] = [round(float(s), 3) for s in sd]                                 # Round the uncertainty standard deviations and append them as a DataFrame column

    # -----------------------------------------------------------------------------------------
    # Step 2 - Real Computational Docking (Oracle)
    # Perform independent rigid/flexible re-docking using tools the agent has never seen.
    # -----------------------------------------------------------------------------------------
    from oracle.dock import which_engine, dock_smiles                                       # Dynamically import docking execution functions from the oracle module
    from oracle.prepare_receptor import ligand_com                                          # Dynamically import the center-of-mass calculator for bounding box definitions
    # Extract the specified docking engine string (e.g., 'gnina', 'smina') from config
    engine = cfg["docking"]["engine"]                                                       
    if which_engine(engine) is not None:                                                    # Verify that the specified docking engine binary is installed and accessible on PATH
        receptor = cfg.get("receptor_pdbqt") or cfg["receptor_pdb"]                         # Prioritize pre-processed PDBQT receptor, falling back to raw PDB
        if not os.path.exists(receptor):                                                    # Check if the resolved receptor file path actually exists on disk
            receptor = cfg["receptor_pdb"]                                                  # Fallback exclusively to the raw PDB path if the PDBQT file is missing
        # Extract docking box bounds from the YAML config, and calculate center 
        # from reference ligand if explicitly missing.
        box = cfg["docking_box"]; center = box["center"] or ligand_com(cfg["ref_ligand_sdf"]) 
        # Extract and cast the spatial bounding box dimensions into a tuple of floats
        size = tuple(box["size_angstrom"])                                                  
        pose_dir = os.path.join(os.path.dirname(args.out) or ".", "poses")                  # Define the directory path to save 3D SDF poses alongside the output CSV
        affs, cnns, scores, poses = [], [], [], []                                          # Initialize empty tracking lists for affinities, CNN affinities, CNN scores, and pose paths
        # Loop through the SMILES strings and perform independent rigid/flexible docking
        for s in smis:                                                                      # Iterate sequentially over every parsed SMILES string in the target batch
            r = dock_smiles(s, receptor, center, size, engine=engine,                       # Execute the heavy docking simulation using the calculated coordinates and engine
                            cnn_scoring=cfg["docking"].get("cnn_scoring", "rescore"),       # Pass the CNN scoring mode setting extracted from the YAML config
                            gpu=cfg["docking"].get("gpu", True), keep_pose_dir=pose_dir)    # Pass GPU acceleration flags and specify where to write the 3D pose files
            affs.append(r.affinity); cnns.append(r.cnn_affinity); scores.append(r.cnn_score); poses.append(r.pose_sdf) # Unpack the docking result object and append metrics and pose to lists
        # Assign the fully populated docking metric lists as columns in the main DataFrame
        # Reminder: affinity = minimizedAffinity / Vina score (kcal/mol, lower=better), 
        # cnn_affinity = gnina CNNaffinity (predicted pKd, higher=better), cnn_score = gnina CNNscore (pose confidence 0..1) 
        df["dock_affinity"] = affs; df["cnn_affinity"] = cnns; df["cnn_score"] = scores     
        
        # -------------------------------------------------------------------------------------
        # Step 3 - PoseBusters Validation
        # Validate the physical realism of the resulting docked poses (e.g., checking clashes).
        # -------------------------------------------------------------------------------------
        try:                                                                                # Wrap PoseBusters execution in a try block to gracefully bypass missing dependencies
            from oracle.posebusters_check import pose_valid_rate                            # Dynamically import the physical sanity checking algorithm
            receptor_pdb = cfg["receptor_pdb"]                                              # Extract the raw unmodified PDB receptor required for clash detection
            valid = []                                                                      # Initialize an empty list to track boolean validation rates for each pose
            # Loop through the 3D SDF pose paths generated by the docking engine
            for p in poses:                                                                 
                if p and os.path.exists(p):                                                 # Verify the string exists and the corresponding file physically resides on disk
                    try:                                                                    # Wrap individual pose checks to prevent single corrupt files from crashing the loop
                        # If the pose path exists, execute the PoseBusters validation
                        # and append the success rate (if it passed or not) to the list
                        rate, _ = pose_valid_rate(p, receptor_pdb); valid.append(rate)      # Execute PoseBusters validation and append the binary float success rate
                    except Exception:                                                       # Catch calculation failures inside the PoseBusters internal physics logic
                        valid.append(np.nan)                                                # Append a NaN placeholder to maintain list length alignment
                else:                                                                       # Execute if the pose path string is empty or the file is missing entirely
                    valid.append(np.nan)                                                    # Append a NaN placeholder to indicate missing physical validation data
            # Create a new DataFrame column from the list of validation rates, calculate 
            # the mean success rate (ignoring NaNs), and print to the console
            df["posebusters_valid"] = valid                                                 # Assign the fully populated list of validation rates as a DataFrame column
            pbv = np.nanmean(valid) if len(valid) else float("nan")                         # Calculate the mean success rate, ignoring NaN gaps, across the entire batch
            print(f"[validate] PoseBusters mean valid fraction: {pbv:.3f}")                 # Print the final physical realism success percentage to the console
        except Exception as e:                                                              # Catch the overarching module import or configuration failure
            print(f"[validate] PoseBusters skipped: {e}")                                   # Print a warning explaining why the PoseBusters stage was bypassed

        # Compute the minimum, median, and mean affinity (estimate of thermodynamic Gibbs free energy of binding in kcal/mol) 
        # across the entire batch of docked poses, and print to the console
        aff_arr = np.array([a for a in affs if a is not None], dtype=float)                 # Convert the python list of affinities to a strict numpy array, dropping null values
        if aff_arr.size:                                                                    # Check if the numpy array contains at least one valid float computation
            print(f"[validate] docking affinity (kcal/mol): "                               # Print the prefix for the statistical summary of the docking simulation
                  f"min {aff_arr.min():.2f}  median {np.median(aff_arr):.2f}  mean {aff_arr.mean():.2f}") # Print the minimum, median, and mean Vina affinity metrics
    # If the user did not request structural docking, skip the docking stage
    else:                                                                                   # Execute if the target docking binary cannot be resolved on the system PATH
        print(f"[validate] no docking engine '{engine}' on PATH — run setup_docking.sh. Skipping docking.") # Alert the user that structural docking was totally bypassed

    # -----------------------------------------------------------------------------------------
    # Step 4 - PhysDock Bridge
    # Trigger ultra-high-fidelity Boltz-2 + OpenMM simulations on the absolute best candidates.
    # -----------------------------------------------------------------------------------------
    # If the user supplied a path pointing to an external PhysDock installation
    if args.physdock:                                                                       # Check if the user supplied a path pointing to an external PhysDock installation
        from oracle.physdock_bridge import run_physics_on_topk                              # Dynamically import the inter-process bridging function to call the physics repository
        # Identify the best column available for sorting candidates (e.g., affinity first, pChEMBL second)
        rank_col = "dock_affinity" if "dock_affinity" in df else ("pred_pchembl" if "pred_pchembl" in df else None) 
        # Sort by ascending kcal/mol or descending pChEMBL, extract top N candidates, 
        # and execute the external PhysDock process (Boltz-2 + OpenMM)
        top = (df.sort_values(rank_col, ascending=(rank_col == "dock_affinity"))["smiles"].head(args.n_physics).tolist() 
               if rank_col else smis[:args.n_physics])                                      # Fallback to slicing the unsorted raw SMILES if neither metric was computed
        info = run_physics_on_topk(top, physdock_repo=args.physdock, conda_env=args.physdock_env) # Execute the external PhysDock process and block until completion
        # Reminder: info = {'ran': True, 'workdir': '/path/to/physdock/output', 'boltz_rc': 0, 'relax_rc': 0, 'note': 'done', 'summary': {...}}
        # Print the returned execution status note from the bridged process
        print(f"[validate] physics: {info['note']}")                                        
    
    # -----------------------------------------------------------------------------------------
    # Data Export
    # Save the consolidated tracking DataFrame to a CSV and print a final status JSON summary.
    # -----------------------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)                            # Create the necessary target parent directories for the CSV, ignoring exists errors
    df.to_csv(args.out, index=False)                                                        # Serialize the Pandas DataFrame and save it locally as a standard CSV format
    print(f"[done] wrote {args.out}")                                                       # Print a notification indicating successful file system write completion
    print(json.dumps({"n": int(len(df)), "columns": list(df.columns)}, indent=2))           # Print a formatted JSON string summarizing row counts and generated columns


if __name__ == "__main__":                                                                  # Prevent automatic execution if this file is imported as a module elsewhere
    main()                                                                                  # Invoke the primary orchestrator function directly to execute the script