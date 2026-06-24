"""
=====================================================
Reference Dataset Generator for Metrics Benchmarking
=====================================================

This script acts as a utility tool within the larger molecular generation pipeline. 
Its primary role is to fetch, clean, and sample a standardized dataset of reference 
molecules (represented as SMILES strings) to compute structural generative metrics 
such as Fréchet ChemNet Distance (FCD), novelty, and scaffold diversity.

It relies on the Therapeutics Data Commons (PyTDC) library to seamlessly download 
and cache large-scale molecular datasets like MOSES (the standard for FCD), ZINC, 
or ChEMBL. The script automatically handles the extraction of the relevant SMILES 
column, deduplication, removal of invalid entries (NaNs), and randomly downsamples 
the set to a manageable, statistically stable size (usually ~30k molecules) before 
saving it as a clean CSV file.

Usage:
    pip install PyTDC
    python make_reference.py --out ../data/ref/chembl_druglike.csv --n 30000 --name MOSES
"""

import argparse
import os


def main():
    """
    Executes the main pipeline for fetching and formatting the reference SMILES dataset.
    
    How it works:
    1. Configures the command-line interface to parse user inputs for output path, sample size, 
       dataset source, and random seed.
    2. Attempts a lazy import of `tdc.generation.MolGen`, failing gracefully with instructions 
       if PyTDC is missing.
    3. Downloads the specified dataset (MOSES, ZINC, or ChEMBL) using PyTDC into a Pandas DataFrame.
    4. Cleans the data by isolating the SMILES column, dropping missing values, and removing duplicates.
    5. Downsamples the dataset to the requested size (`n`) using a fixed seed for reproducibility.
    6. Creates the necessary nested directories and exports the final dataset to a CSV file.
    
    Args:
        None (Arguments are parsed via `argparse` from `sys.argv`).
        
        CLI Arguments:
        --out (str): Required. The file path where the final CSV will be saved.
        --n (int): The target number of molecules to sample (default: 30000).
        --name (str): The TDC dataset to pull ["MOSES", "ZINC", "ChEMBL"] (default: "MOSES").
        --seed (int): The random seed for the sampling operation (default: 42).
        
    Returns:
        None.
        
    Example:
        $ python make_reference.py --out ./data/moses_ref.csv --n 10000 --name MOSES
        Downloading/loading MOSES via TDC (first run caches to ./data)...
        Wrote 10000 reference SMILES -> ./data/moses_ref.csv
    """
    # --------------------------------------------------------------------------------------
    # Command Line Interface Setup
    # Configures and parses command-line arguments to dictate the output path, sample size,
    # reference dataset source, and random seed.
    # --------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the argument parser object to handle command-line inputs
    ap.add_argument("--out", required=True)                                                 # Define the required output file path parameter for the resulting CSV
    ap.add_argument("--n", type=int, default=30000, help="sample size (FCD is stable from ~10k)") # Define the sample size parameter, defaulting to 30k for stable statistical metrics
    ap.add_argument("--name", default="MOSES", choices=["MOSES", "ZINC", "ChEMBL"])         # Define the source dataset parameter, strictly limiting choices to known generative benchmarks
    ap.add_argument("--seed", type=int, default=42)                                         # Define the random seed parameter to ensure deterministic sampling across runs
    args = ap.parse_args()                                                                  # Parse the supplied command-line arguments and store them in the 'args' namespace

    # ---------------------------------------------------------------------------------------
    # Dependency Validation
    # Attempt to load the required TDC library, providing a clean fallback if it is missing.
    # ---------------------------------------------------------------------------------------
    try:                                                                                    # Wrap the import statement in a try block to gracefully catch missing optional dependencies
        from tdc.generation import MolGen                                                   # Import the molecular generation dataset loader from the PyTDC library
    except Exception as e:                                                                  # Catch any import errors or module-not-found exceptions
        raise SystemExit(                                                                   # Abort execution immediately and print a helpful error message to the terminal
            "PyTDC not installed. Run `pip install PyTDC`, or supply your own reference "   # Provide exact instructions on how the user can resolve the missing dependency
            f"CSV with a 'smiles' column instead.\n(import error: {e})"                     # Append the specific Python exception message for debugging clarity
        )                                                                                   # Close the SystemExit exception call

    # ---------------------------------------------------------------------------------------
    # Data Fetching and Cleaning
    # Download the benchmark dataset, isolate the SMILES strings, remove bad/duplicate data, 
    # and downsample to the requested size ('n') using a fixed seed for reproducibility.
    # ---------------------------------------------------------------------------------------
    print(f"Downloading/loading {args.name} via TDC (first run caches to ./data)...")       # Notify the user that the download has started and may take a moment if not cached
    df = MolGen(name=args.name).get_data()                                                  # Request the dataset from PyTDC, which returns a Pandas DataFrame containing the chemical data
    col = "smiles" if "smiles" in df.columns else df.columns[0]                             # Identify the target column: look for "smiles" explicitly, otherwise fallback to the first column
    df = df[[col]].dropna().drop_duplicates()                                               # Subset the dataframe to just the SMILES column, remove empty rows, and strip out duplicate molecules
    if len(df) > args.n:                                                                    # Check if the cleaned dataset size exceeds the user's requested sample size
        df = df.sample(n=args.n, random_state=args.seed)                                    # Randomly subsample the dataframe down to exactly 'n' rows using the fixed seed
    df = df.rename(columns={col: "smiles"})                                                 # Standardize the dataframe column name to strictly be "smiles" for downstream metric scripts

    # ---------------------------------------------------------------------------------------
    # File Output
    # Ensure the destination directory exists and save the cleaned dataframe to disk.
    # ---------------------------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)                  # Dynamically create the parent directory structure for the output file if it doesn't already exist
    df.to_csv(args.out, index=False)                                                        # Export the standardized Pandas DataFrame to a CSV file without writing the integer index
    print(f"Wrote {len(df)} reference SMILES -> {args.out}")                                # Print a final success message summarizing the number of molecules saved and the file location


if __name__ == "__main__":
    main()                                                                                  # Invoke the main pipeline execution function