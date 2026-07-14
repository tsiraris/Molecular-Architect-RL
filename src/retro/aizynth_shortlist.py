"""
============================================
Stage 3 Retrosynthesis Shortlist Evaluation
============================================

This script executes real retrosynthesis planning on the final shortlist of generated molecules.
Throughout earlier stages, the Synthetic Accessibility (SA) score is used as a fast heuristic proxy 
to estimate synthesizability. This script elevates that claim from a simple heuristic descriptor to a 
demonstrated, multi-step synthetic pathway.

It uses the open-source retrosynthesis planner, AiZynthFinder, to search for viable multi-step routes 
leading back to commercially purchasable building blocks. The script reports the overall fraction of 
the shortlist that has a solved route and exports detailed statistics per molecule.

Note: This module requires `aizynthfinder` to be installed (`pip install aizynthfinder`), along 
with its necessary configuration, stock, and policy files (as detailed in AiZynthFinder docs). 
The script is designed to "fail open" (returning None) if the dependency or configuration is missing, 
ensuring it never blocks or crashes the overarching pipeline.
"""

import argparse
import json
from typing import List, Optional


def route_solved(smiles: List[str], config_yml: str) -> Optional[dict]:
    """
    Evaluates a list of SMILES strings to determine if viable retrosynthetic routes exist.
    
    Safely attempts to load the AiZynthFinder module. If successful, it initializes the finder 
    with the provided YAML configuration, selects "zinc" as the stock (purchasable building blocks), 
    and "uspto" for both expansion and filter policies (reaction rules). It then iterates through 
    the provided SMILES, runs a tree search for each, builds the routes, and extracts statistics 
    (whether it was solved, number of distinct routes, and number of steps).
    
    Args:
        smiles (List[str]): A list of target molecule SMILES strings to evaluate.
        config_yml (str): The file path to the AiZynthFinder configuration YAML file.
        
    Returns:
        Optional[dict]: A dictionary containing the overall 'fraction_solved' and a nested 
        'per_molecule' dictionary with route statistics. Returns None if AiZynthFinder is missing.
        
    Example:
        >>> smis = ["CC(=O)Oc1ccccc1C(=O)O"]
        >>> stats = route_solved(smis, "config.yml")
        >>> print(stats["fraction_solved"])
        1.0
    """
    # -------------------------------------------------------------------------------------
    # Dependency Check & Initialization
    # Attempt to load the retrosynthesis engine and safely fail open if unavailable.
    # -------------------------------------------------------------------------------------
    try:                                                                                    # Wrap the import statement to catch missing environment dependencies gracefully
        from aizynthfinder.aizynthfinder import AiZynthFinder                               # Import the primary AiZynthFinder class required for retrosynthetic tree searches
    except Exception:                                                                       # Catch any exception raised if the package is not installed in the environment
        print("[retro] aizynthfinder not installed -> skipping (pip install aizynthfinder).") # Print a warning message to the console advising the user of the skipped step
        return None                                                                         # Return None to fail open, allowing the broader pipeline to continue unimpeded
        
    finder = AiZynthFinder(configfile=config_yml)                                           # Instantiate the AiZynthFinder object using the specified YAML configuration file
    
    # -------------------------------------------------------------------------------------
    # Database & Policy Configuration: Select the chemical databases for building blocks 
    # (ZINC for purchasable stock) and reaction rule policies (USPTO).
    # -------------------------------------------------------------------------------------
    finder.stock.select("zinc"); finder.expansion_policy.select("uspto"); finder.filter_policy.select("uspto") # Configure the finder to use ZINC for purchasable stock and USPTO for reaction rules
    
    out = {}                                                                                # Initialize an empty dictionary to accumulate the resulting statistics for each molecule
    
    # -------------------------------------------------------------------------------------
    # Tree Search Execution
    # Iterate through target molecules, compute routes, and extract outcome statistics.
    # -------------------------------------------------------------------------------------
    for smi in smiles:                                                                      # Iterate sequentially through the provided list of target SMILES strings
        finder.target_smiles = smi                                                          # Set the current SMILES string as the active target for the retrosynthesis engine
        finder.tree_search(); finder.build_routes()                                         # Execute the internal tree search algorithm and subsequently compile the discovered routes
        stats = finder.extract_statistics()                                                 # Extract the dictionary of performance and outcome statistics for the current tree search
        out[smi] = {"solved": bool(stats.get("is_solved", False)),                          # Record a boolean flag indicating if at least one complete route was found
                    "n_routes": int(stats.get("number_of_routes", 0)),                      # Record the total integer count of valid synthetic routes discovered
                    "n_steps": stats.get("number_of_steps")}                                # Record the expected number of synthetic steps required for the routes
                    
    # -------------------------------------------------------------------------------------
    # Results Aggregation
    # Compute the macro-level success rate across the entire provided batch.
    # -------------------------------------------------------------------------------------
    n_solved = sum(1 for v in out.values() if v["solved"])                                  # Calculate the total number of molecules that successfully yielded a valid synthetic route
    return {"fraction_solved": n_solved / max(1, len(smiles)), "per_molecule": out}         # Return a compiled dictionary containing the overall success fraction and the granular per-molecule data


def main():
    """
    Command-line interface entry point for executing shortlist retrosynthesis.
    
    Initializes an argument parser to collect the target SMILES file path, the configuration 
    YAML path, and the desired JSON output path. Reads the SMILES file, delegates execution 
    to `route_solved`, and dumps the resulting data to the specified JSON file. Prints a 
    summary percentage to standard output.
    
    Args:
        None.
        
    Returns:
        None.
    """
    # -------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Define and collect runtime arguments for file inputs and outputs.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the argument parser to handle command-line inputs
    ap.add_argument("--smiles_file", required=True)                                         # Define a mandatory argument for the path to the text file containing target SMILES
    ap.add_argument("--config", required=True, help="aizynthfinder config.yml")             # Define a mandatory argument for the path to the AiZynthFinder YAML configuration
    ap.add_argument("--out", default="../results/retro.json")                               # Define an optional argument for the output JSON path, defaulting to a specific relative path
    args = ap.parse_args()                                                                  # Parse the supplied command-line arguments into a populated namespace object
    
    # -------------------------------------------------------------------------------------
    # Data I/O and Execution
    # Load targets, run the analysis, save the results, and print a summary.
    # -------------------------------------------------------------------------------------
    smis = [l.strip() for l in open(args.smiles_file) if l.strip()]                         # Read the input file line-by-line, stripping whitespace and ignoring empty lines to build a SMILES list
    res = route_solved(smis, args.config)                                                   # Invoke the route solver function on the loaded SMILES list using the provided configuration
    json.dump(res, open(args.out, "w"), indent=2)                                           # Serialize the resulting dictionary to a JSON file at the specified output path with 2-space indentation
    print(f"[retro] {res['fraction_solved']*100:.0f}% solved" if res else "[retro] unavailable") # Print the formatted percentage of solved routes to the console, or a fallback message if unavailable


if __name__ == "__main__":                                                                  # Ensure the main function only triggers if the script is executed directly, not imported
    main()                                                                                  # Invoke the primary CLI entry point function