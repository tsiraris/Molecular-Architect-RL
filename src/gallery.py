"""
=============================
Evaluation Gallery Generator
=============================

This script serves as a visual evaluation tool for the molecular generation pipeline.
It takes a text file containing generated SMILES strings (typically the top-K molecules 
from a specific checkpoint) and renders them into an annotated 2D image grid. 

Crucially, it labels each molecule with its Quantitative Estimate of Drug-likeness (QED),
its Synthetic Accessibility (SA) score, and the definitive verdict from the `synth_gate` 
(whether the structure is physically realistic or contains reward-hacking topological exploits).
This allows researchers to visually compare the outputs of different reward functions 
(e.g., an unguarded QED-only reward vs. a synthesizability-gated reward).

Usage (run from the `src/` directory):
    python gallery.py --gen ../results/topk_OLD.txt --out ../results/gallery_old.png --title "Before (QED-only reward)"
    python gallery.py --gen ../results/topk_NEW.txt --out ../results/gallery_new.png --title "After (gated reward)"
"""

import argparse
import os
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, QED
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")                                                              # Silence RDKit's C++ standard error logs to keep the console output clean

# Dynamically adjust the Python path to ensure that the reward gating modules can be imported regardless of the current working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))                                 # Dynamically append this file's own directory (src/) to the Python path so the reward/ and eval/ sub-packages resolve
try:                                                                                        # Attempt to import the reward gating modules assuming a standard project structure
    from reward.synth_gate import synth_gate, sa_score                                      # Import the synthetic reality check function and the SA scoring metric
except Exception:                                                                           # Catch ImportErrors if the script is run from a flattened or non-standard directory
    from synth_gate import synth_gate, sa_score                                             # Fallback import: assume the gating modules live in the exact same directory

# Dynamically adjust the Python path to ensure that the metrics utility can be imported regardless of the current working directory
try:                                                                                        # Attempt to import the file-reading utility from the evaluation module
    from eval.metrics import read_smiles                                                    # Import the specific function designed to safely parse SMILES text files
except Exception:                                                                           # Catch ImportErrors if standard directory structure fails
    from metrics import read_smiles                                                    # Fallback import: assume the metrics utility lives in the same directory


def build_legend(mol: Chem.Mol) -> str:
    """
    Calculates properties and constructs the annotation string for a single molecule.
    
    Evaluates the RDKit Mol object through the QED heuristic, the SA scorer, and the 
    custom `synth_gate`. It extracts the rejection reason if the molecule is banned, 
    and formats these metrics into a concise string suitable for the image grid.
    
    Args:
        mol (Chem.Mol): The RDKit molecule object to be evaluated.
        
    Returns:
        str: The formatted legend string (e.g., "QED 0.85 | SA 2.4 | BANNED:cumulene").
        
    Example:
        >>> m = Chem.MolFromSmiles("C#S#S#C")   # contains a banned cumulated triple-bond motif
        >>> build_legend(m)
        'QED 0.42 | SA 6.5 | BANNED:banned_motif'
    """
    q = QED.qed(mol)                                                                        # Calculate the raw Quantitative Estimate of Drug-likeness (QED) score [0,1]
    ok, soft, info = synth_gate(mol)                                                        # Pass the molecule through the strict topological reality gate to catch reward hacks
    verdict = "OK" if ok else f"BANNED:{info.get('reason','?')}"                            # Set verdict to "OK" if passed, otherwise extract the specific rejection reason from the info dict
    return f"QED {q:.2f} | SA {sa_score(mol):.1f} | {verdict}"                              # Format and return the final string containing QED, rounded SA score, and the gate verdict


def main():
    """
    Main execution block: parses CLI arguments, processes SMILES, and renders the gallery.
    
    Sets up the argument parser to accept input/output paths and formatting preferences.
    Reads the top-K SMILES from the specified text file, converts them to valid RDKit Mol 
    objects, builds their respective legends, and leverages RDKit's `MolsToGridImage` 
    to render the final PNG. Lastly, it prints a brief statistical summary to stdout.
    """
    # -------------------------------------------------------------------------------------
    # CLI Parsing
    # Define and parse the command-line arguments needed to run the script.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Instantiate the standard Python command-line argument parser
    ap.add_argument("--gen", required=True)                                                 # Define the mandatory '--gen' argument pointing to the input text file of generated SMILES
    ap.add_argument("--out", required=True)                                                 # Define the mandatory '--out' argument specifying the destination filepath for the rendered PNG
    ap.add_argument("--title", default="")                                                  # Define an optional '--title' argument to annotate the console output summary
    ap.add_argument("--n", type=int, default=12, help="max molecules to show")              # Define '--n' to cap the number of molecules drawn (defaults to 12)
    ap.add_argument("--cols", type=int, default=4)                                          # Define '--cols' to specify the number of columns in the rendered image grid (defaults to 4)
    args = ap.parse_args()                                                                  # Parse the supplied command-line arguments into the 'args' namespace object

    # -------------------------------------------------------------------------------------
    # Data Ingestion & Object Instantiation
    # Parse the valid SMILES of the top-K molecules, convert them to RDKit Mol objects, 
    # and build the paired list of molecule objects and their image grid legends.
    # -------------------------------------------------------------------------------------
    smis = read_smiles(args.gen)[: args.n]                                                  # Read the SMILES file using the imported utility and truncate the list to the requested max 'n'
    mols, legends = [], []                                                                  # Initialize two empty lists to store the successfully parsed RDKit Mol objects and their string labels
    for s in smis:                                                                          # Iterate over each raw SMILES string loaded from the target file
        m = Chem.MolFromSmiles(s)                                                           # Attempt to parse the SMILES string into an RDKit Mol object
        if m is None:                                                                       # Check if RDKit failed to parse the string (resulting in a None object)
            continue                                                                        # Skip this specific SMILES string and proceed to the next iteration if parsing failed
        mols.append(m)                                                                      # Append the successfully instantiated RDKit Mol object to the collection list
        legends.append(build_legend(m))                                                     # Generate the evaluation string for this molecule and append it to the legends list

    # -------------------------------------------------------------------------------------
    # Image Rendering & Disk I/O
    # Ensure data exists, render the 2D grid via RDKit, save the file, and log summary.
    # -------------------------------------------------------------------------------------
    if not mols:                                                                            # Evaluate if the molecule list is entirely empty after the parsing loop
        print("No valid molecules to draw.")                                                # Print a warning to the console indicating the failure to find valid data
        return                                                                              # Abort the main function execution gracefully

    img = Draw.MolsToGridImage(                                                             # Call RDKit's built-in grid rendering utility to create the composite image
        mols, molsPerRow=args.cols, subImgSize=(320, 260),                                  # Pass the Mol list, set the column count, and define the pixel dimensions of each individual cell
        legends=legends, useSVG=False,                                                      # Provide the parallel list of text labels, and enforce raster (PNG) rendering instead of vector (SVG)
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)                  # Ensure the target output directory exists by creating all missing parent directories automatically
    img.save(args.out)                                                                      # Write the rendered grid image out to the disk at the path specified by '--out'
    n_banned = sum("BANNED" in lg for lg in legends)                                        # Calculate the total count of molecules that failed the synthesizability gate by scanning the legends
    print(f"Wrote {args.out}  ({len(mols)} mols, {n_banned} flagged BANNED)  {args.title}") # Print a final execution summary to the console including output path, valid count, ban count, and title


if __name__ == "__main__":                                                                  # Check if this Python script is being executed directly (as opposed to being imported as a module)
    main()                                                                                  # Trigger the primary execution flow of the application