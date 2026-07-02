"""
========================
Docking Label Generator
========================

This script is responsible for the creation of the docking dataset, that will complement 
the real-but-sparse pChEMBL dataset in build_training_set.py, to train the surrogate model 
to operate within the RL pipeline. In specific, the script docks a diverse drug-like library 
against a defined protein target (e.g., the KRAS G12C box) to produce dense, self-generated 
docking affinity labels. 

How it works:
1. Reads a Library: It takes a CSV file containing a diverse library of generic drug-like molecules 
                   (SMILES strings) and optionally subsamples them (e.g., down to 8,000 to save time).
                   
2. Prepares the Target: It loads the YAML configuration to locate the 3D protein receptor (KRAS G12C)
                        and calculates the exact spatial boundaries (docking box) where the drug should bind.
                        
3. Runs Batch Docking: It dispatches the molecules to a physics engine (gnina, smina, or vina). The 
                       engine converts the 2D SMILES into 3D structures, simulates how they fit into 
                       the protein pocket, and calculates a physics-based and a CNN-based binding score.
                       
4. Exports the Labels: It extracts the successful physics scores (kcal/mol, where lower is better) 
                       and CNN-based scores, saving them into a new file (default: docking_labels.csv).


Output:
A CSV file containing the columns: smiles, dock_affinity (kcal/mol, lower=better), 
cnn_affinity, and cnn_score.

Performance Note:
Throughput depends heavily on the engine and hardware. For example, using gnina GPU 
on an A10G processes approximately 1–3 ligands/s when using `cnn_scoring=rescore`. 
At this rate, 8–10k molecules take a few hours; 50k takes overnight. It is recommended 
to start with `--n 8000` for the first surrogate, then scale later.

Prerequisites:
Run `oracle/prepare_receptor.py` and `setup_docking.sh` prior to executing this script.
"""
import argparse
import os
import sys

import pandas as pd
import yaml

# -----------------------------------------------------------------------------------------
# Path Configuration & Imports
# Dynamically link the relative path to ensure custom modules can be imported correctly.
# -----------------------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))                    # Inject the parent src/ directory into the system path to allow local imports
from oracle.dock import dock_many, which_engine                                             # noqa: E402
from oracle.prepare_receptor import ligand_com                                              # noqa: E402


def _read_library(path, n, seed=42):
    """
    Reads a chemical library from a CSV file, extracts the SMILES column, and optionally subsamples it.
    
    Loads the target CSV using Pandas, dynamically searches for a column named 'smiles' or 
    'canonical_smiles' (falling back to the first column if neither is found). It drops any 
    missing values and converts them to a list of strings. If a subsample size `n` is provided 
    that is smaller than the dataset, it randomly shuffles the list (using a fixed seed for 
    reproducibility) and truncates it.
    
    Args:
        path (str): The file path to the target CSV chemical library.
        n (int): The maximum number of molecules to extract (0 or None means all).
        seed (int, optional): The random seed for reproducible shuffling. Defaults to 42.
        
    Returns:
        List[str]: A list of valid SMILES strings ready for the docking pipeline.
        
    Example:
        >>> smis = _read_library("../data/druglike_library.csv", n=8000)
        >>> len(smis)
        8000
    """
    # -------------------------------------------------------------------------------------------
    # Data Loading & Column Resolution: Load the file, isolate the primary string representations
    # of the molecules, drop any missing entries, and convert to a Python list of strings.
    # -------------------------------------------------------------------------------------------
    df = pd.read_csv(path)                                                                  # Load the tabular CSV file containing the chemical library into a pandas DataFrame
    col = next((c for c in df.columns if c.lower() in ("smiles", "canonical_smiles")), df.columns[0]) # Dynamically identify the SMILES column by searching for 'smiles' or 'canonical_smiles', defaulting to the first column
    smis = df[col].dropna().astype(str).tolist()                                            # Extract the identified column, drop missing values, cast to strings, and convert to a native Python list
    
    # -----------------------------------------------------------------------------------------------------
    # Subsampling Logic: If a specific subsample size `n` is requested and the library exceeds it,
    # shuffle the list using a fixed random seed for reproducibility and truncate it to the requested size.
    # -----------------------------------------------------------------------------------------------------
    if n and len(smis) > n:                                                                 # Check if a specific subsample size was requested and if the library exceeds this capacity
        import random                                                                       # Import the random module locally to avoid global namespace pollution
        random.Random(seed).shuffle(smis)                                                   # Initialize a seeded random number generator and shuffle the SMILES list in-place for reproducibility
        smis = smis[:n]                                                                     # Truncate the shuffled list to explicitly return only the requested number of molecules
    return smis                                                                             # Return the final curated list of SMILES strings


def main():
    """
    Main execution routine for the Docking Label Generator script.
    
    Step-by-Step:
    1. Parses Command Line Arguments (CLI) to accept config files, library paths, and run parameters.
    2. Loads the target YAML configuration specifying the receptor, docking bounding box, and engine constraints.
    3. Resolves the spatial coordinates of the target binding pocket (either static from config or computed via ligand center-of-mass).
    4. Validates that the requested docking engine (gnina, smina, vina) is installed and available on the system PATH.
    5. Reads and subsamples the raw chemical library.
    6. Dispatches the batch docking task using the `dock_many` parallel orchestrator.
    7. Parses the structured outputs and exports the successful structural affinities to a dense label CSV.
    
    Args:
        None (Uses sys.argv implicitly via argparse).
        
    Returns:
        None. Writes data strictly to disk.
    """
    # -------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Define and collect the required input parameters to orchestrate the script.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the argument parser to handle command line inputs
    ap.add_argument("--config", required=True)                                              # Require a YAML configuration file path outlining the receptor and docking parameters
    ap.add_argument("--library", required=True, help="CSV with a smiles column (drug-like library)") # Require the target input CSV library of drug-like molecules to be evaluated
    ap.add_argument("--out", default="../data/docking_labels.csv")                          # Define the destination path for the output labels CSV, setting a safe default
    # Subsample size: 0 = all, otherwise limit to n molecules for faster execution
    ap.add_argument("--n", type=int, default=8000, help="subsample size (0 = all)")         # Define the integer subsample size to cap computation, defaulting to 8000 as a baseline
    ap.add_argument("--engine", default=None, help="override config engine (gnina|smina|vina)") # Allow manual CLI override of the backend docking engine specified in the YAML config
    args = ap.parse_args()                                                                  # Parse and extract all provided arguments from the system execution string

    # -------------------------------------------------------------------------------------
    # Configuration & Target Resolution: Load the YAML settings, resolve the receptor, the 
    # target's bounding box, and the docking engine (validating if it is installed).
    # -------------------------------------------------------------------------------------
    cfg = yaml.safe_load(open(args.config))                                                 # Open and parse the provided YAML configuration file into a standard python dictionary
    receptor = cfg.get("receptor_pdbqt") or cfg["receptor_pdb"]                             # Attempt to retrieve the pre-processed PDBQT receptor, falling back to raw PDB
    if not os.path.exists(receptor):                                                        # Verify if the prioritized receptor file actually exists on the filesystem
        receptor = cfg["receptor_pdb"]                                                      # Fallback strictly to the unformatted PDB file if the PDBQT version is missing entirely
    box = cfg["docking_box"]                                                                # Extract the bounding box configuration dictionary outlining the physical docking search space
    center = box["center"] or ligand_com(cfg["ref_ligand_sdf"])                             # Resolve the center coordinate: use config if present, otherwise dynamically compute the center of mass of the reference ligand
    size = tuple(box["size_angstrom"])                                                      # Extract and cast the bounding box dimensions into a tuple representing physical Angstroms
    engine = args.engine or cfg["docking"]["engine"]                                        # Determine the active docking engine, allowing CLI arguments to override the YAML configuration
    if which_engine(engine) is None:                                                        # Validate that the selected docking engine executable is actually accessible within the system PATH
        print(f"[labels] No docking engine '{engine}' on PATH. Run setup_docking.sh first.", file=sys.stderr) # Print a fatal error message guiding the user to run the environment setup script
        sys.exit(1)                                                                         # Forcefully terminate the script with a standard non-zero error code

    # -------------------------------------------------------------------------------------
    # Docking Execution: Load the SMILES of the molecules from the diverse library, and
    # execute the massive batch docking task across all of the queued molecules.
    # -------------------------------------------------------------------------------------
    smis = _read_library(args.library, args.n)                                              # Load and potentially subsample the target SMILES sequence library from disk
    print(f"[labels] docking {len(smis)} molecules with {engine} against {receptor} @ {center}") # Output execution summary logging the library size, target engine, receptor file, and spatial center
    results = dock_many(smis, receptor, center, size, engine=engine,                        # Execute the massive batch docking orchestrator across all queued SMILES sequences
                        # rescore to get CNN scores ("cnn_affinity" and "cnn_score") 
                        # in addition to the physics-based docking score ("dock_affinity")
                        cnn_scoring=cfg["docking"].get("cnn_scoring", "rescore"),           # Forward the neural scoring instruction (e.g., 'rescore') to the engine if using gnina
                        gpu=cfg["docking"].get("gpu", True))                                # Forward the hardware acceleration flag allowing the engine to leverage CUDA

    # ------------------------------------------------------------------------------------------------
    # Result Processing & Export: Filter the successful docking runs, build a structured DataFrame 
    # of the results, write it to a CSV file, and print the range of the computed physical affinities.
    # ------------------------------------------------------------------------------------------------
    rows = [(r.smiles, r.affinity, r.cnn_affinity, r.cnn_score) for r in results if r.ok]   # Extract the string representation and calculated physical affinities strictly for docking runs that completed successfully
    df = pd.DataFrame(rows, columns=["smiles", "dock_affinity", "cnn_affinity", "cnn_score"]) # Package the successful structural docking outputs into a structured pandas DataFrame
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)                            # Ensure the target output directory exists, silently creating missing parent folders if necessary
    df.to_csv(args.out, index=False)                                                        # Serialize and export the final DataFrame to a CSV file on disk, excluding the arbitrary integer row index
    print(f"[labels] wrote {len(df)}/{len(smis)} successful docks -> {args.out}")           # Log the absolute success rate mapping the number of completed docks to the initial library size
    if len(df):                                                                             # Verify that at least one molecule successfully generated valid docking calculations
        print(f"[labels] dock_affinity range {df['dock_affinity'].min():.2f}..{df['dock_affinity'].max():.2f} " # Print the lower and upper bounds of the computed physical affinities
              f"(kcal/mol, lower=better)")                                                  # Remind the user of the unit scale where highly negative kcal/mol indicates tighter binding

if __name__ == "__main__":                                                                  # Prevent automatic execution of main routine when imported as an external module
    main()                                                                                  # Formally invoke the master script logic sequence