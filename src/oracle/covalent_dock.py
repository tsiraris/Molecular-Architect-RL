"""
==================================
Covalent Docking Oracle (Stage 3)
==================================

This module provides the physics-based scoring mechanism for the Stage 3 target-aware 
reinforcement learning pipeline, specifically targeting KRAS G12C. 

Unlike Stage 2 which relied on non-covalent physics, this oracle correctly models the 
covalent bond formation between the ligand's electrophilic warhead and the nucleophilic 
sulfur (SG) of the Cys12 residue on the KRAS protein.

How it works:
It acts as a filtering and execution gateway. First, it strictly screens incoming molecules 
to ensure they possess a valid Michael-acceptor warhead. Molecules lacking this feature 
are immediately rejected (and can be routed to the standard non-covalent path). 
Eligible molecules are embedded in 3D and passed to the `gnina` docking engine using 
specialized covalent flags (`--covalent_rec_atom`, `--covalent_lig_atom_pattern`, 
`--covalent_optimize_lig`). The resulting covalent complex is then scored using gnina's 
convolutional neural network (CNN) to extract minimized affinity and CNN scores. 
It seamlessly returns a `DockResult` dataclass, maintaining API uniformity with the 
non-covalent oracle.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from typing import List, Optional

from rdkit import Chem

from oracle.dock import DockResult, which_engine, embed3d, _parse_sdf, _parse_stdout
from reward.warhead import has_warhead

# -----------------------------------------------------------------------------------------
# Covalent Warhead Definition
# Define the specific functional group required for a molecule to be eligible for 
# covalent docking against the target residue.
# -----------------------------------------------------------------------------------------
_LIG_WARHEAD_SMARTS = "[CH2]=[CH]C(=O)N"                                                    # The ligand atom pattern that forms the covalent bond: the terminal CH2 of an acrylamide Michael acceptor.


def covalent_dock_smiles(smi: str, receptor: str, cys_spec: str = "A:12:SG",
                         lig_smarts: str = _LIG_WARHEAD_SMARTS, engine: str = "gnina",
                         cnn_scoring: str = "rescore", timeout: int = 600,
                         keep_pose_dir: Optional[str] = None) -> DockResult:
    """
    Executes covalent docking for a single warhead-bearing molecule against a specified receptor.
    
    Initializes a `DockResult` object that will be populated with "minimized affinity","CNNaffinity", 
    and "CNNscore" metrics from gnina. Verifies the molecule can be parsed and contains the required warhead. 
    Checks that the correct docking engine (`gnina`) is available. Generates a 3D conformation of
    the ligand, writes it to a temporary SDF, and invokes `gnina` via a subprocess with covalent 
    optimization flags. Parses the CNN scoring output and optionally saves the resulting 3D pose to disk.
    Returns a `DockResult` dataclass for the ligand containing its SMILES, the docking success flag, gnina metrics 
    ("minimized affinity", "CNNaffinity", "CNNscore"), and pose paths (optional).
    
    Args:
        smi (str): The SMILES string of the ligand to be docked.
        receptor (str): The file path to the receptor structure (e.g., PDB or PDBQT file).
        cys_spec (str, optional): The chain, residue number, and atom name of the target cysteine. Defaults to "A:12:SG".
        lig_smarts (str, optional): SMARTS pattern identifying the attacking atom pattern on the ligand. Defaults to _LIG_WARHEAD_SMARTS.
        engine (str, optional): The docking engine to use; must be "gnina" for covalent mode. Defaults to "gnina".
        cnn_scoring (str, optional): The gnina CNN scoring mode (e.g., "rescore"). Defaults to "rescore".
        timeout (int, optional): Maximum execution time in seconds before killing the subprocess. Defaults to 600.
        keep_pose_dir (Optional[str], optional): Directory path to save successful docked pose SDFs. Defaults to None.
        
    Returns:
        DockResult: A dataclass containing the ligand SMILES, docking success flag, gnina metrics 
        ("minimized affinity", "CNNaffinity", "CNNscore"), and pose paths (optional).
        
    Example:
        >>> smi = "C=CC(=O)Nc1ccccc1" # Example acrylamide
        >>> result = covalent_dock_smiles(smi, "kras_g12c_rec.pdb")
        >>> print(result.ok)
        True
    """
    # -------------------------------------------------------------------------------------
    # Pre-Docking Validation & Setup
    # Validate the chemical structure, ensure warhead presence, and prepare 3D coordinates.
    # -------------------------------------------------------------------------------------
    res = DockResult(smiles=smi)                                                            # Initialize the standardized result container pre-loaded with the query SMILES
    mol = Chem.MolFromSmiles(smi)                                                           # Convert the input raw SMILES string into an RDKit Mol topological object
    if mol is None or not has_warhead(mol):                                                 # Check if RDKit failed to parse the molecule or if it lacks the required Michael-acceptor warhead
        return res                                                                          # Immediately return the failed default result object as it is not covalent-eligible
    exe = which_engine(engine)                                                              # Resolve the absolute system path to the requested docking engine executable
    if exe is None or exe != "gnina":                                                       # Verify the engine exists and is explicitly gnina, as this covalent mode is gnina-specific
        return res                                                                          # Abort and return the default failed result if the correct engine is unavailable
    
    m3d = embed3d(smi)                                                                      # Generate a minimized 3D conformation of the ligand using standard embedding heuristics
    if m3d is None:                                                                         # Check if the 3D embedding process failed (e.g., due to severe steric clashes)
        return res                                                                          # Return the failed result object since docking requires 3D coordinates
    
    tmp = tempfile.mkdtemp()                                                                # Create a secure, isolated temporary directory on the OS filesystem for staging I/O files
    try:                                                                                    # Wrap the filesystem and subprocess operations in a try block to ensure cleanup
        
        # ---------------------------------------------------------------------------------
        # I/O Staging and Engine Execution
        # Write the ligand SDF to disk, build the command list with covalent flags, and run.
        # ---------------------------------------------------------------------------------
        lig = os.path.join(tmp, "lig.sdf"); w = Chem.SDWriter(lig); w.write(m3d); w.close() # Define ligand path, open an RDKit SDWriter, serialize the 3D molecule to disk, and flush/close the writer
        out = os.path.join(tmp, "out.sdf")                                                  # Define the exact target file path where the docking engine should write the scored poses
        cmd = [exe, "-r", receptor, "-l", lig, "-o", out,                                   # Construct the base command list: executable, receptor flag, ligand flag, and output flag
               "--covalent_rec_atom", cys_spec,                                             # Append the gnina flag explicitly defining the receptor's target atom (e.g., Cys12 SG)
               "--covalent_lig_atom_pattern", lig_smarts,                                   # Append the gnina flag providing the SMARTS pattern to identify the attacking ligand atom
               "--covalent_optimize_lig",                                                   # Append the gnina flag commanding it to physically model the covalent bond geometry
               "--cnn_scoring", cnn_scoring, "--seed", "0", "--num_modes", "3"]             # Append flags for CNN rescoring, deterministic random seeding, and limiting the output to top 3 modes
        
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)            # Execute the constructed command synchronously, capturing stdout/stderr as strings, enforcing the timeout
        
        # ---------------------------------------------------------------------------------
        # Output Parsing and Pose Archiving
        # Extract gnina metrics from the output files and save successful poses.
        # ---------------------------------------------------------------------------------
        # Parse the generated output SDF file to extract CNN affinity, CNN score, and minimized energies
        res = _parse_sdf(out)                                                               
        if not res.ok:                                                                      # Check if the SDF parsing failed (usually implies the docking engine crashed or found no poses)
            res = _parse_stdout(r.stdout + "\n" + r.stderr)                                 # Attempt a fallback parsing directly from the engine's standard output and error streams
        res.smiles = smi                                                                    # Re-inject the original query SMILES string into the populated result object
        
        if res.ok and keep_pose_dir:                                                        # Archive the successful pose metrics SDF if requested in the user-specified directory
            os.makedirs(keep_pose_dir, exist_ok=True)                                       # Ensure the destination archive directory physically exists, creating it if necessary
            dst = os.path.join(keep_pose_dir, f"cov_{abs(hash(smi)) % (10**8)}.sdf")        # Construct a unique destination filename by hashing the SMILES string
            try:                                                                            # Wrap the file copy operation in an inner try block to avoid failing the whole run on I/O issues
                shutil.copy(out, dst); res.pose_sdf = dst                                   # Copy the scored output SDF to the permanent archive and record the path in the result object
            except Exception:                                                               # Catch any operating system errors during the file copying process
                pass                                                                        # Silently ignore the copy failure and proceed to return the numerical results
        # Return the fully populated DockResult dataclass containing scores and statuses
        return res                                                                          
    except Exception:                                                                       # Catch any broad exceptions, including subprocess timeouts or severe system failures
        # Return the baseline failed result object to prevent pipeline crashes
        return res                                                                          
    finally:                                                                                # Ensure this block executes regardless of success, failure, or early returns
        # ---------------------------------------------------------------------------------
        # Resource Cleanup
        # Recursively delete the temporary staging directory to prevent disk bloat.
        # ---------------------------------------------------------------------------------
        shutil.rmtree(tmp, ignore_errors=True)                                              # Force-remove the temporary directory and all its contents, ignoring missing file errors


def covalent_dock_many(smiles: List[str], receptor: str, cys_spec: str = "A:12:SG",
                       engine: str = "gnina", cnn_scoring: str = "rescore",
                       keep_pose_dir: Optional[str] = None, progress: bool = True,
                       cache=None) -> List[DockResult]:
    """
    Batch processes a list of SMILES strings for covalent docking against a receptor.
    
    Iterates over a provided list of SMILES. Optionally wraps the iterable in a `tqdm` 
    progress bar for CLI feedback. Calls `covalent_dock_smiles` for each molecule, 
    aggregating the `DockResult` objects into a list.

    Stage-3 acceleration: covalent docking is gnina-only and shares the single GPU, so unlike the 
    non-covalent path it is not fanned across processes (stacking gnina on one GPU only thrashes
    VRAM). Instead it consults an optional disk cache first — the covalent warhead set is small and
    heavily re-encountered across acquisition rounds and resumed sessions, so caching removes almost
    all redundant covalent docks at zero cost to accuracy (every dock is still seeded identically).
    
    Args:
        smiles (List[str]): A list of SMILES strings to be docked.
        receptor (str): The file path to the receptor structure.
        cys_spec (str, optional): Target cysteine specification. Defaults to "A:12:SG".
        engine (str, optional): Docking engine. Defaults to "gnina".
        cnn_scoring (str, optional): CNN scoring mode. Defaults to "rescore".
        keep_pose_dir (Optional[str], optional): Directory to save poses. Defaults to None.
        progress (bool, optional): Whether to display a tqdm progress bar. Defaults to True.
        cache (Optional[DockCache], optional): A disk cache to consult/populate. Defaults to None.
        
    Returns:
        List[DockResult]: A list of resulting "DockResult" dataclasses corresponding to the input SMILES order.
        
    Example:
        >>> batch = ["C=CC(=O)N", "c1ccccc1"]
        >>> results = covalent_dock_many(batch, "rec.pdb", progress=False)
        >>> len(results)
        2
    """
    # -------------------------------------------------------------------------------------
    # Cache Pass
    # Serve any covalent dock already computed under these parameters straight from disk.
    # -------------------------------------------------------------------------------------
    key = dict(receptor=receptor, cys_spec=cys_spec, engine=engine, covalent=True)          # Score-affecting key: covalent=True keeps these distinct from non-covalent cache rows
    results: List[Optional[DockResult]] = [None] * len(smiles)                              # Pre-allocate an order-preserving result slot per input molecule
    todo = []                                                                               # Collect (index, smiles) pairs that miss the cache and must actually be docked
    # Resolve cache hits (previously-computed identical covalent docks) 
    # up front so gnina is only ever invoked on genuinely-new warheads
    for i, s in enumerate(smiles):                                                          
        hit = cache.get(s, **key) if cache is not None else None                            # Probe the disk cache (returns None when caching is disabled or on a miss)
        if hit is not None:                                                                 # A hit is a previously-computed identical covalent dock
            results[i] = DockResult(**hit)                                                  # Rehydrate the stored dict back into a DockResult dataclass
        # Otherwise queue the molecule for a real gnina covalent dock
        else:                                                                               
            todo.append((i, s))                                                             # Remember its original index so the fresh result slots back in order

    # -------------------------------------------------------------------------------------
    # Compute Pass (sequential, gnina/GPU-bound)
    # Setup optional progress tracking and dock each cache-miss on the shared GPU in turn.
    # -------------------------------------------------------------------------------------
    # Iterate only over the cache-misses that genuinely require docking
    it = todo                                                                               
    # If the user requested a progress bar and there are actually molecules to dock, 
    # wrap the iterable in tqdm
    if progress and todo:                                                                   # Only construct a progress bar when there is real docking work to display
        try:                                                                                # Wrap the import in a try block in case the tqdm library is not installed
            from tqdm import tqdm; it = tqdm(todo, desc="covalent-dock")                    # Import tqdm locally and wrap the cache-miss iterable with a labeled progress bar
        except Exception:                                                                   # Catch the import error if tqdm is missing
            pass                                                                            # Silently ignore the error and continue with the standard list iterable
    # Iterate over the cache-misses, docking each one sequentially on the shared GPU, 
    # and storing the results in their original order
    for (i, s) in it:                                                                       # Dock each remaining warhead-bearing molecule one at a time on the GPU
        res = covalent_dock_smiles(s, receptor, cys_spec, engine=engine, cnn_scoring=cnn_scoring, # Invoke the unchanged single-molecule covalent docking pipeline
                                   keep_pose_dir=keep_pose_dir)                             # Pass the common configuration arguments through verbatim
        results[i] = res                                                                    # Store the result in its order-preserving slot
        # If a disk cache is provided and the docking was successful, 
        # persist the result to disk for future reuse
        if cache is not None and res.ok:                                                    # Persist only successful covalent docks so failures can be retried on the next pass
            cache.put(s, asdict(res), **key)                                                # Stage-3 accel: serialise DockResult -> dict for the disk cache (free repeats/resumes)
    # Return the list of results, preserving the original input order and filling any gaps with default failure records
    return [r if r is not None else DockResult(smiles=smiles[i]) for i, r in enumerate(results)] 