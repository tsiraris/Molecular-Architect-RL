"""
============================
Oracle Receptor Preparation
============================

This script functions as the primary one-time setup utility for preparing biological 
targets (receptors) for the 3D docking oracle pipeline. Before the reinforcement learning 
agent can be rewarded based on active-learning docking scores (e.g., against KRAS G12C), 
the raw protein structure must be formatted, and the 3D search space defined.

Role & Capabilities:
1. Receptor Preparation: Automatically converts a cleaned PhysDock PDB receptor file 
   into the PDBQT format required by docking engines (like Vina/Gnina) using Meeko.
2. Search Space Definition: Computes the geometric center of the docking box by 
   calculating the center of mass (unweighted heavy-atom centroid) of a known reference 
   co-crystal ligand. 

Usage:
Run this script manually from the `src/` directory prior to launching the RL training loop:
    python -m oracle.prepare_receptor --config ../configs/kras_g12c.yaml

The script will write the generated PDBQT file next to the original receptor and output 
the calculated docking box center coordinates, which can then be permanently pinned 
in the YAML configuration file for reproducible RL runs.
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
import yaml
from rdkit import Chem


def ligand_com(sdf_path: str):
    """
    Calculates the center of mass (unweighted heavy-atom centroid) of a reference ligand.
    
    Loads a structure-data file (SDF) using RDKit, extracting the first valid molecule. 
    It iterates over all atoms in the stored 3D conformer, extracts their (x, y, z) 
    Cartesian coordinates into a NumPy array, and calculates the arithmetic mean across 
    the atomic axis. This centroid becomes the focal point for the docking box.
    
    Args:
        sdf_path (str): The absolute or relative file path to the reference ligand SDF.
        
    Returns:
        List[float]: A list of three floats representing the [x, y, z] coordinates of the centroid.
        
    Raises:
        ValueError: If the SDF file cannot be read or contains no valid RDKit molecules.
        
    Example:
        >>> # Assuming 'kras_ligand.sdf' exists and is a valid 3D structure
        >>> center = ligand_com('kras_ligand.sdf')
        >>> print(center)
        [15.231, -8.442, 22.109]
    """
    # ---------------------------------------------------------------------------------------------
    # Supplier Initialization
    # Load the reference ligand SDF file and extract the first valid RDKit molecule representation.
    # ---------------------------------------------------------------------------------------------
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)                                           # Initialize an RDKit SDMolSupplier to read the ligand file, keeping hydrogen atoms intact
    mol = next((m for m in suppl if m is not None), None)                                          # Iterate through the supplier and extract the first successfully parsed RDKit molecule object
    if mol is None:                                                                                # Check if the parsing failed entirely or yielded a null/invalid molecule
        raise ValueError(f"could not read ligand from {sdf_path}")                                 # Raise a descriptive ValueError halting execution if the ligand geometry cannot be loaded
    
    # --------------------------------------------------------------------------------------------------------
    # Coordinate Extraction & Center Calculation: Retrieve the 3D conformer in the parsed molecule, extract 
    # atomic positions for all atoms into a 2D NumPy array, and compute the unweighted arithmetic mean (COM).
    # --------------------------------------------------------------------------------------------------------
    conf = mol.GetConformer()                                                                      # Retrieve the default 3D conformer generated or stored within the parsed molecule
    pts = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])              # Extract the (x, y, z) coordinates for every atom index and cast them into a 2D NumPy array
    return pts.mean(axis=0).tolist()                                                               # Calculate the unweighted arithmetic mean across all atom coordinates and return as a Python list


def prepare_receptor_pdbqt(receptor_pdb: str, out_pdbqt: str) -> bool:
    """
    Converts a standard PDB protein file into a docking-ready PDBQT file using Meeko.
    
    Searches the system PATH for the Meeko `mk_prepare_receptor` executable. If found, 
    it dispatches a subprocess command passing the input PDB and the desired output 
    PDBQT paths. It captures execution streams and verifies the output file was written.
    
    Args:
        receptor_pdb (str): File path to the cleaned input protein structure (.pdb).
        out_pdbqt (str): Destination file path for the generated output structure (.pdbqt).
        
    Returns:
        bool: True if the conversion succeeded and the file exists, False otherwise.
        
    Example:
        >>> success = prepare_receptor_pdbqt('receptor.pdb', 'receptor.pdbqt')
        >>> print(success)
        True
    """
    # -----------------------------------------------------------------------------------------
    # Executable Resolution
    # Check the system PATH for the Meeko preparation script (evaluating both naming conventions).
    # -----------------------------------------------------------------------------------------
    if shutil.which("mk_prepare_receptor.py") is None and shutil.which("mk_prepare_receptor") is None: # Evaluate if neither variant of the Meeko executable exists in the system environment path
        print("[prepare_receptor] Meeko not found (pip install meeko). Skipping PDBQT generation.")    # Print a warning message to the console advising the user to install the Meeko dependency
        return False                                                                                   # Abort the preparation process and return False to formally indicate operational failure
    exe = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")                # Assign the first successfully resolved Meeko executable path string to the 'exe' variable
    
    # -----------------------------------------------------------------------------------------
    # Subprocess Execution
    # Execute the Meeko binary against the input PDB to generate the formatted output PDBQT.
    # -----------------------------------------------------------------------------------------
    try:                                                                                               # Wrap the subprocess call in a try-except block to gracefully catch critical execution failures
        subprocess.run([exe, "--read_pdb", receptor_pdb, "-o", out_pdbqt, "-p"],                       # Construct and dispatch the command line invocation passing the input, output, and parameters
                       check=True, capture_output=True, text=True)                                     # Enforce strict execution checking, capture stdout/stderr streams, and decode to text strings
        return os.path.exists(out_pdbqt)                                                               # Verify the requested output file was actually written to disk and return the boolean status
    except Exception as e:                                                                             # Catch any OS, pathing, or execution errors thrown during the subprocess pipeline run
        print(f"[prepare_receptor] Meeko failed ({e}). gnina can also read a plain PDB receptor;"      # Print a detailed error message notifying the user of the failure and explaining a potential fallback
              f" set receptor_pdbqt to the .pdb if needed.")                                           # Suggest modifying the configuration YAML to bypass the rigid PDBQT requirement
        return False                                                                                   # Return False signifying the explicit PDBQT generation procedure crashed or was interrupted


def main():
    """
    Main execution wrapper bridging command line arguments with the preparation routines.
    
    Parses the mandatory `--config` argument pointing to a YAML configuration file. 
    It inspects the `docking_box` parameters to determine if a center must be dynamically 
    calculated from a reference SDF or parsed directly. It prints these coordinates for 
    the user, and then invokes the PDBQT generation utility for the defined receptor.
    
    Args:
        None (Consumes arguments from `sys.argv` via `argparse`).
        
    Returns:
        None
        
    Example:
        $ python prepare_receptor.py --config target.yaml
        [prepare_receptor] box center = ['10.000', '15.000', '-5.000']  size = 20.0
        ...
    """
    # -----------------------------------------------------------------------------------------
    # Configuration Parsing
    # Parse command line arguments and securely load the target YAML configuration file.
    # -----------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                                     # Instantiate an ArgumentParser object to formally handle command line user inputs
    ap.add_argument("--config", required=True)                                                         # Define a mandatory '--config' flag requiring the path to the pipeline YAML configuration file
    args = ap.parse_args()                                                                             # Parse the provided command line arguments into a structured python namespace object
    cfg = yaml.safe_load(open(args.config))                                                            # Open and safely deserialize the specified YAML configuration file into a native Python dictionary

    # -----------------------------------------------------------------------------------------
    # Docking Box Resolution
    # Determine the center of the docking box dynamically via reference ligand or static config.
    # -----------------------------------------------------------------------------------------
    box = cfg["docking_box"]                                                                           # Extract the 'docking_box' sub-dictionary from the parsed overarching configuration
    # If the config dictates deriving the center from a ligand AND lacks a hardcoded center
    if box.get("center_from") == "ref_ligand" and not box.get("center"):                               # Check if the config dictates deriving the center from a ligand AND lacks a hardcoded center
        # Dynamically compute the volumetric center of mass of the reference ligand
        center = ligand_com(cfg["ref_ligand_sdf"])                                                     # Dynamically compute the volumetric center of mass of the reference ligand using the provided reference SDF file path
    else:                                                                                              # Execute this branch if the user provided hardcoded coordinates or chose an alternative method
        # Assign the hardcoded central coordinates directly from the loaded configuration dictionary
        center = box["center"]                                                                         # Assign the hardcoded central coordinates directly from the loaded configuration dictionary
    print(f"[prepare_receptor] box center = {['%.3f' % c for c in center]}  size = {box['size_angstrom']}") # Print the newly resolved docking box center coordinates and pre-configured size to the console
    print("[prepare_receptor] paste into configs docking_box.center to pin it.")                       # Advise the user to physically hardcode the computed coordinates for future reproducibility

    # -----------------------------------------------------------------------------------------
    # Receptor Formatting
    # Trigger the Meeko preparation pipeline using the validated configuration file paths.
    # -----------------------------------------------------------------------------------------
    ok = prepare_receptor_pdbqt(cfg["receptor_pdb"], cfg["receptor_pdbqt"])                            # Attempt to execute the PDBQT generation pipeline and store the resulting boolean success flag
    print(f"[prepare_receptor] receptor PDBQT: {'written ' + cfg['receptor_pdbqt'] if ok else 'not written (use the PDB directly with gnina)'}") # Print a final status message indicating explicit success or suggesting a plain-text gnina PDB fallback

if __name__ == "__main__":                                                                             # Shield internal executable blocks from unintended firing during external module imports
    main()                                                                                             # Formally invoke the main configuration procedure to initialize the receptor setup