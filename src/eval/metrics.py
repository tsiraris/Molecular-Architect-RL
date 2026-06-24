"""
============================================================
Generative-Model Benchmarks (GuacaMol/MOSES Style)
============================================================

This script evaluates the quality of a de novo molecular generative model by calculating
a comprehensive suite of standard cheminformatics metrics. It serves as a rapid diagnostic 
tool to ensure the generative model produces structurally valid, diverse, and synthesizable 
molecules before advancing to target-aware 3D docking.

Core metrics calculated (which run instantly without a reference set):
- Validity: Fraction of generated SMILES that parse as valid chemical graphs.
- Uniqueness: Fraction of valid molecules that are non-duplicates (canonicalized).
- Internal Diversity: Average pairwise Tanimoto distance of Morgan fingerprints.
- Scaffold Diversity: Fraction of unique Bemis-Murcko scaffolds among valid molecules.
- SA Score: Synthetic Accessibility statistics (mean, 90th percentile, fraction <= 4).

Reference-dependent metrics (require a reference dataset like ChEMBL/ZINC):
- Novelty: Fraction of valid generated molecules not present in the reference set.
- FCD (Frechet ChemNet Distance): Distributional similarity between generated and 
  reference molecules (requires 'fcd_torch' package).

The script is highly flexible, natively parsing raw `.smi` files, CSVs, and custom 
tabular outputs from the RL agent.
"""

import argparse
import os
import sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")                                                              # Disable all RDKit application-level logging and C++ warnings

# -----------------------------------------------------------------------------------------
# Optional Dependencies Loading
# Attempt to load RDKit's SA Score module and the optional FCD neural network metric.
# -----------------------------------------------------------------------------------------
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))                            # Append the RDKit community contributions directory to the system path
import sascorer  # noqa: E402

try:                                                                                        # Wrap the FCD import in a try-except block to make it entirely optional
    from fcd_torch import FCD                                                               # Import the Frechet ChemNet Distance module for distributional metrics
    _HAS_FCD = True                                                                         # Set a global flag indicating FCD is installed and available
except Exception:                                                                           # Catch ImportErrors or any other initialization failures from fcd_torch
    _HAS_FCD = False                                                                        # Set the global flag to False to gracefully skip FCD computation


#---------------------------------------- I/O -----------------------------------------
def read_smiles(path):
    """
    Parses a file and extracts a list of SMILES strings.
    
    Employs basic heuristic sniffing to determine the file structure. It supports 
    custom TopK output formats (pipe-separated with headers), standard CSV files 
    (detecting a 'smiles' column), and raw `.smi` or `.txt` files (one SMILES per line).
    
    Args:
        path (str): The absolute or relative file path to the molecule dataset.
        
    Returns:
        list: A list of extracted raw SMILES strings.
        
    Example:
        >>> smis = read_smiles("data/generated.csv")
        >>> len(smis)
        1000
    """
    # Open the file and strip out empty lines to prepare for formatting heuristics.
    smis = []                                                                               # Initialize an empty list to accumulate the extracted SMILES strings
    with open(path) as f:                                                                   # Open the designated file path in standard read mode
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]                                 # Extract all non-empty lines from the file and strip trailing newlines
    if not lines:                                                                           # Check if the file was entirely empty or composed only of whitespace
        return smis                                                                         # Return the empty SMILES list immediately

    # Handle the specific pipe-delimited format exported by the RL training loop.
    if "|" in lines[0] and "smiles" in lines[0].lower():                                    # Detect if the first line is a pipe-separated header containing 'smiles'
        header = [h.strip().lower() for h in lines[0].split("|")]                           # Split the header string by pipes, strip whitespace, and convert to lowercase
        sidx = header.index("smiles")                                                       # Find the integer column index where the SMILES strings are located
        for ln in lines[1:]:                                                                # Iterate over all subsequent data rows in the parsed file
            if set(ln) <= set("-| "):                                                       # Check if the line is purely a markdown-style separator rule (dashes, pipes)
                continue                                                                    # Skip parsing this separator line
            cells = [c.strip() for c in ln.split("|")]                                      # Split the data row by pipes and strip whitespace from every cell
            if len(cells) > sidx:                                                           # Verify that the row actually has enough columns to contain a SMILES string
                smis.append(cells[sidx])                                                    # Extract the SMILES string from its specific column and append to the list
        return smis                                                                         # Return the populated list of SMILES strings for this format

    # Handle standard comma-separated values, searching for a 'smiles' column header.
    if path.lower().endswith(".csv") and ("smiles" in lines[0].lower() or "," in lines[0]): # Detect CSV extension and check for either a 'smiles' header or basic commas
        header = [h.strip().lower() for h in lines[0].split(",")]                           # Split the header row by commas and normalize to lowercase
        sidx = header.index("smiles") if "smiles" in header else 0                          # Default to the first column (index 0) if no explicit 'smiles' header is found
        for ln in lines[1:]:                                                                # Loop through all the data rows following the header
            cells = ln.split(",")                                                           # Split the current row string into discrete cells by commas
            if len(cells) > sidx:                                                           # Ensure the row isn't malformed and possesses the required column index
                smis.append(cells[sidx].strip())                                            # Isolate the SMILES string, strip whitespace, and append it
        return smis                                                                         # Return the parsed SMILES list for the CSV format

    # Fallback to assuming the first whitespace-separated token on each line is a SMILES.
    for ln in lines:                                                                        # Iterate through every non-empty line in the file
        smis.append(ln.split()[0])                                                          # Extract the first whitespace-separated token as the SMILES string
    return smis                                                                             # Return the accumulated list of SMILES strings


# ------------------------------------------------------------ metrics ---------
def _mols(smis):
    """
    Converts a list of SMILES strings into RDKit Mol objects.
    
    Iterates through the provided list and attempts to parse each SMILES string.
    Invalid chemical strings will return `None` from the RDKit parser.
    
    Args:
        smis (list of str): A list of raw SMILES strings.
        
    Returns:
        list: A list of RDKit Mol objects, where invalid inputs are represented as None.
        
    Example:
        >>> _mols(["C", "INVALID"])
        [<rdkit.Chem.rdchem.Mol object>, None]
    """
    # Generate and return a list of RDKit Mol objects (or Nones) from the SMILES strings
    return [Chem.MolFromSmiles(s) for s in smis]                                            


def validity(smis):
    """
    Calculates the proportion of structurally valid generated molecules.
    
    Attempts to convert all input SMILES to RDKit Mol objects. It filters out any 
    `None` values (which indicate parsing failures) and computes the ratio of valid 
    molecules to the total input size.
    
    Args:
        smis (list of str): A list of raw SMILES strings produced by the generator.
        
    Returns:
        tuple: A tuple containing (1) the list of valid RDKit Mol objects, and 
        (2) the float validity ratio [0.0, 1.0].
        
    Example:
        >>> valid_mols, score = validity(["C", "CC", "C1C"]) # C1C is invalid
        >>> print(score)
        0.6666666666666666
    """
    # Validity Metric: Convert SMILES to Mol objects and compute the fraction that are valid.
    mols = _mols(smis)                                                                      # Delegate to the helper function to batch-convert strings into Mol objects
    valid = [m for m in mols if m is not None]                                              # Create a filtered list retaining only successfully parsed Mol objects
    return valid, len(valid) / max(1, len(smis))                                            # Return the valid mols list and the fraction of inputs that were valid


def uniqueness(valid_mols):
    """
    Calculates the proportion of unique molecules among the valid generations.
    
    Converts all valid RDKit Mol objects back into strictly canonicalized SMILES 
    strings. Puts them in a Python `set` to remove duplicates and divides the set 
    size by the total list size.
    
    Args:
        valid_mols (list of Chem.Mol): A list of valid RDKit Mol objects.
        
    Returns:
        float: The uniqueness ratio [0.0, 1.0].
        
    Example:
        >>> mols = [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C")]
        >>> uniqueness(mols)
        0.5
    """
    # Uniqueness Metric: Map valid graphs to canonical strings and measure set uniqueness.
    canon = [Chem.MolToSmiles(m) for m in valid_mols]                                       # Re-export each Mol object as a strictly canonicalized SMILES string
    return len(set(canon)) / max(1, len(canon))                                             # Deduplicate the canonical strings via a set and divide by the total count


def novelty(valid_mols, ref_smis):
    """
    Calculates the proportion of generated molecules not found in a reference dataset.
    
    Canonicalizes the reference dataset into a fast lookup `set`. It then checks 
    each canonicalized generated molecule against this set. Novelty is the fraction 
    of generated molecules that do not hit the reference set.
    
    Args:
        valid_mols (list of Chem.Mol): A list of valid generated RDKit Mol objects.
        ref_smis (list of str): A list of raw SMILES strings from the training/reference set.
        
    Returns:
        float: The novelty ratio [0.0, 1.0]. Returns 0.0 if no valid molecules exist.
        
    Example:
        >>> gen = [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")]
        >>> novelty(gen, ["C"])
        0.5
    """
    # Novelty Metric: Hash reference molecules into a set and check generated molecules against it.
    ref = set()                                                                             # Initialize an empty set to hold canonical reference SMILES for O(1) lookups
    for s in ref_smis:                                                                      # Iterate through every raw reference SMILES string
        m = Chem.MolFromSmiles(s)                                                           # Attempt to parse the reference SMILES into an RDKit Mol object
        if m: ref.add(Chem.MolToSmiles(m))                                                  # If parsing succeeded, add its standardized canonical SMILES to the reference set
    gen = [Chem.MolToSmiles(m) for m in valid_mols]                                         # Extract standard canonical SMILES for all valid generated molecules
    if not gen: return 0.0                                                                  # Prevent division by zero if the generative model produced absolutely no valid outputs
    return sum(g not in ref for g in gen) / len(gen)                                        # Count how many generated strings bypass the reference set and divide by the total


def internal_diversity(valid_mols):
    """
    Calculates the internal diversity of the generated molecule set.
    
    Computes Morgan fingerprints (radius 2, 2048 bits) for all valid molecules. 
    It then calculates the pairwise Tanimoto similarity for every unique pair 
    in the set. Diversity is defined as (1 - average_similarity).
    
    Args:
        valid_mols (list of Chem.Mol): A list of valid RDKit Mol objects.
        
    Returns:
        float: The internal diversity score [0.0, 1.0]. Higher is more diverse.
        
    Example:
        >>> mols = [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")]
        >>> div = internal_diversity(mols)
    """
    # Internal Diversity Metric: Extract structural fingerprints, calculate aggregate pairwise distance, and return the 1-mean_diversity.
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]           # Compute 2048-bit Morgan (ECFP4) structural fingerprints for all valid molecules
    if len(fps) < 2: return 0.0                                                             # Immediately return zero diversity if there are fewer than two items to compare
    sims = []                                                                               # Initialize an empty list to accumulate pairwise similarity scores
    for i in range(len(fps)):                                                               # Iterate through the list of fingerprints using an integer index
        sims += list(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))               # Compute bulk Tanimoto similarities for all subsequent unique pairs to avoid double-counting
    return 1.0 - float(np.mean(sims)) if sims else 0.0                                      # Subtract the mean pairwise similarity from 1.0 to obtain standard distance/diversity


def scaffold_diversity(valid_mols):
    """
    Calculates the structural scaffold diversity of the generated molecules.
    
    Extracts the core Bemis-Murcko scaffold (rings and connecting linkers, removing 
    sidechains) for each valid molecule. Diversity is the ratio of unique scaffolds 
    to the total number of valid molecules.
    
    Args:
        valid_mols (list of Chem.Mol): A list of valid RDKit Mol objects.
        
    Returns:
        float: The scaffold diversity ratio [0.0, 1.0].
        
    Example:
        >>> mols = [Chem.MolFromSmiles("c1ccccc1C"), Chem.MolFromSmiles("c1ccccc1O")]
        >>> scaffold_diversity(mols) # Both reduce to bare benzene
        0.5
    """
    # Scaffold Diversity Metric: Extract backbone scaffolds to assess higher-level structural 
    # variety, and return the ratio of unique scaffolds to the total.
    scafs = set()                                                                           # Initialize an empty set to accumulate unique Bemis-Murcko scaffold strings
    for m in valid_mols:                                                                    # Iterate through each valid RDKit molecule object
        try: scafs.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))                          # Attempt to extract the molecular backbone and add its SMILES to the set
        except Exception: pass                                                              # Ignore severe ring parsing failures triggered by topologically distorted graphs
    return len(scafs) / max(1, len(valid_mols))                                             # Divide the final count of unique scaffolds by the total generated batch size


def sa_stats(valid_mols):
    """
    Computes statistical metrics for the Synthetic Accessibility (SA) Score.
    
    Evaluates the RDKit Contrib SA Score for each valid molecule, which estimates 
    synthesizability on a scale of 1 (easy) to 10 (hard). Returns a dictionary 
    containing the mean score, the 90th percentile (worst-case), and the fraction 
    of molecules considered plausibly synthesizable (score <= 4.0).
    
    Args:
        valid_mols (list of Chem.Mol): A list of valid RDKit Mol objects.
        
    Returns:
        dict: A dictionary containing 'sa_mean', 'sa_p90', and 'sa_frac_le_4'.
        
    Example:
        >>> stats = sa_stats([Chem.MolFromSmiles("C")])
        >>> print(list(stats.keys()))
        ['sa_mean', 'sa_p90', 'sa_frac_le_4']
    """
    # Synthetic Accessibility Assessment: Extract SA scores for each valid molecule and compute 
    # aggregate statistics such as mean, 90th percentile, and fraction below the standard threshold of 4.0.
    sas = []                                                                                # Initialize an empty list to store the numerical Synthetic Accessibility scores
    for m in valid_mols:                                                                    # Iterate over all topologically valid graph representations
        try: sas.append(float(sascorer.calculateScore(m)))                                  # Compute the SA score using historical fragment counts and append it
        except Exception: pass                                                              # Silently bypass any extreme valence or hypergraph failures during SA calculation
    if not sas: return {"sa_mean": None, "sa_p90": None, "sa_frac_le_4": None}              # Return a dictionary of Nones if no valid scores could be successfully extracted
    sas = np.array(sas)                                                                     # Convert the accumulated raw scores into a NumPy array for efficient stats aggregation
    return {"sa_mean": float(sas.mean()),                                                   # Calculate and store the overall mathematical average of the SA scores
            "sa_p90": float(np.percentile(sas, 90)),                                        # Calculate the 90th percentile SA score (the value that 90% of the generated molecules have an SA score equal to or better (lower) than)
            "sa_frac_le_4": float((sas <= 4.0).mean())}                                     # Compute the percentage of the batch that falls under the standard feasibility threshold


def report(gen_smis, ref_smis=None, device="cpu"):
    """
    Generates a comprehensive benchmark report dictionary from raw SMILES.
    
    Executes the entire suite of metrics. It runs the baseline independent 
    metrics first. If reference SMILES are provided, it computes novelty, and 
    if 'fcd_torch' is installed, it computes the distributional Frechet ChemNet Distance.
    
    Args:
        gen_smis (list of str): A list of generated SMILES strings.
        ref_smis (list of str, optional): Reference SMILES for relative metrics. Defaults to None.
        device (str, optional): Compute device for FCD neural network ('cpu' or 'cuda'). Defaults to "cpu".
        
    Returns:
        dict: A compiled dictionary of metric names mapped to rounded float values or strings.
        
    Example:
        >>> rep = report(["C", "CC"], ref_smis=["C"])
        >>> rep["validity"]
        1.0
    """
    # Run the core metrics that do not require external calibration data.
    valid_mols, val = validity(gen_smis)                                                    # Parse graphs and extract the absolute validity ratio immediately
    out = {                                                                                 # Initialize the overarching benchmark report dictionary
        "n_generated": len(gen_smis),                                                       # Log the raw scale of the generative attempt
        "validity": round(val, 4),                                                          # Record validity rounded to four decimal places for clean formatting
        "uniqueness": round(uniqueness(valid_mols), 4),                                     # Execute uniqueness check and record to four decimal places
        "internal_diversity": round(internal_diversity(valid_mols), 4),                     # Execute pairwise fingerprint diversity and record to four decimal places
        "scaffold_diversity": round(scaffold_diversity(valid_mols), 4),                     # Execute Bemis-Murcko backbone diversity and record
    }
    out.update({k: (round(v, 4) if v is not None else None) for k, v in sa_stats(valid_mols).items()}) # Merge SA score stats directly into the report, handling None edge-cases

    # Run distributional and novelty checks, such as FCD and novelty, only if a reference database was provided.
    if ref_smis:                                                                            # Verify if the user supplied a baseline reference dataset
        out["novelty"] = round(novelty(valid_mols, ref_smis), 4)                            # Compute reference-bypass ratio and attach it to the report dictionary
        if _HAS_FCD:                                                                        # Verify if the heavy PyTorch dependency loaded successfully at script start
            try:                                                                            # Protect the deep learning forward pass from breaking the reporting sequence
                gen_canon = [Chem.MolToSmiles(m) for m in valid_mols]                       # Standardize generative strings required by the FCD neural backend
                ref_canon = [s for s in ref_smis if Chem.MolFromSmiles(s)]                  # Parse and standardize reference strings (purging invalid reference rows)
                out["fcd"] = round(float(FCD(device=device)(gen_canon, ref_canon)), 4)      # Initialize the FCD model on the specified hardware and compute distance
            except Exception as e:                                                          # Catch CUDA memory errors or mismatched torch tensor problems
                out["fcd"] = f"error: {e}"                                                  # Append a localized error message string instead of crashing the process
        else:                                                                               # Handle the case where the external FCD package is missing entirely
            out["fcd"] = "skipped (pip install fcd_torch)"                                  # Inject an informative fallback string into the output dictionary
    return out                                                                              # Return the finalized dictionary mapping all computed benchmarks


def main():
    """
    Parses execution arguments from standard input (`--gen`, `--ref`, `--device`). 
    Reads the associated files, triggers the report compilation, and cleanly 
    prints the resulting dictionary to standard output in an aligned key-value format.
    """
    # Command Line Interface Setup
    ap = argparse.ArgumentParser()                                                                  # Instantiate an argument parser to handle standard shell arguments
    ap.add_argument("--gen", required=True, help="generated molecules: topk .txt / .csv / .smi")    # Register the required flag pointing to the RL agent's output dump
    ap.add_argument("--ref", default=None, help="reference SMILES for novelty/FCD (.csv/.smi)")     # Register an optional flag pointing to baseline chemistry (e.g., ZINC)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (FCD only)")                       # Register an optional compute device override flag
    args = ap.parse_args()                                                                          # Extract the requested configurations from sys.argv execution

    
    # Run data parsing on both the generated and reference files, perform benchmark calculation,
    # and display results cleanly to console.
    gen = read_smiles(args.gen)                                                             # Parse the specified target generation file using dynamic format sniffing
    ref = read_smiles(args.ref) if args.ref else None                                       # Only attempt to parse the reference file if the explicit argument was provided
    res = report(gen, ref, device=args.device)                                              # Construct the overarching benchmark report dict against requested hardware
    width = max(len(k) for k in res)                                                        # Find the longest metric name key to align the output columns cleanly
    print(f"\n=== Stage 1: metrics for {args.gen} ===")                                     # Print a formatted string header indicating which file was evaluated
    for k, v in res.items():                                                                # Loop through every extracted metric pair within the dictionary
        print(f"  {k:<{width}} : {v}")                                                      # Print left-aligned metric names padded dynamically, followed by the computed score
    print()                                                                                 # Print a final blank line for aesthetic terminal spacing


if __name__ == "__main__":                                                                  # Verify that the script is being executed natively and not imported elsewhere
    main()                                                                                  # Fire the main CLI execution loop