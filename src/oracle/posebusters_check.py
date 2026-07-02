"""
=====================================================
Stage-2 Oracle — PoseBusters Physical-Validity Check
=====================================================

This script executes the PoseBusters physical-validity suite on generated docked poses.
It serves as a critical results-table component for the target-aware Phase-2 pipeline, 
acting as a strict filter against unrealistic generated geometries.

PoseBusters subjects the docked protein-ligand complexes to a rigorous battery of tests, 
evaluating structural integrity metrics such as internal steric strain, proper planarity, 
ideal bond lengths and angles, and checking for severe protein-ligand clashes. The fraction 
of poses passing all of these checks ("% PoseBusters-valid") is used as a strong, current 
results metric. 

Note on dependencies: This module explicitly relies on the external `posebusters` library. 
If it is not installed (`pip install posebusters`), the script will raise a clear, hard 
error at runtime. This prevents silent degradation of the evaluation pipeline, ensuring 
that the generated results columns are always mathematically and chemically rigorous.
"""
from typing import Optional, Tuple

# -----------------------------------------------------------------------------------------
# Core PoseBusters Evaluation
# Wraps the external PoseBusters library to run physical validity tests on molecule files.
# -----------------------------------------------------------------------------------------
def bust(docked_sdf: str, receptor_pdb: str, cond: str = "dock"):
    """
    Executes the comprehensive PoseBusters evaluation suite on docked ligand poses.
    
    Performs a delayed (lazy) import of the `PoseBusters` class to initialize the evaluation engine 
    with a specific configuration condition (e.g., standard docking checks). It then runs the `bust` 
    method against the provided ligand SDF and receptor PDB files, checking for clashes, strain, 
    and geometric distortions.
    
    Args:
        docked_sdf (str): The file path to the SDF containing the generated/docked 3D ligand poses.
        receptor_pdb (str): The file path to the PDB containing the target protein receptor structure.
        cond (str, optional): The configuration preset for PoseBusters. Defaults to "dock".
        
    Returns:
        pandas.DataFrame: A detailed DataFrame containing boolean test results for every check 
        across every evaluated pose, such as: "clash", "strain", "planarity", "bond", and "angle".
        
    Example:
        >>> df_results = bust("ligands_docked.sdf", "kras_g12c.pdb")
        >>> print(df_results.columns)
    """
    from posebusters import PoseBusters                                                     # Perform a delayed import to ensure 'posebusters' is only required when this specific oracle is invoked, raising a clear error if missing (no silent degradation).
    # Initialize the PoseBusters engine with the config='dock' preset
    pb = PoseBusters(config=cond)                                                           # Initialize the PoseBusters test engine using the provided configuration condition (defaulting to "dock" for docking constraints).
    # Execute the evaluation suite against the ligand SDF and receptor PDB 
    # (no known native ligand reference provided via None), returning the raw DataFrame.
    # The DataFrame contains boolean results (True/False) for each of the 5 PoseBusters 
    # checks ("clash", "strain", "planarity", "bond", and "angle"), whether the pose 
    # passed each check or not.
    return pb.bust(docked_sdf, None, receptor_pdb)                                          


# -----------------------------------------------------------------------------------------
# Pass-Rate Aggregation
# Computes the strict aggregate metric (% PoseBusters-valid) required for results tracking.
# -----------------------------------------------------------------------------------------
def pose_valid_rate(docked_sdf: str, receptor_pdb: str) -> Tuple[float, Optional[object]]:
    """
    Calculates the absolute pass-rate of docked poses against the PoseBusters physical constraints.
    
    Invokes the core `bust` function to retrieve the raw results DataFrame. It then reduces 
    the DataFrame horizontally (across all checks) to determine if a pose is perfectly valid, 
    and vertically (across all poses) to compute the fractional pass rate.
    
    Args:
        docked_sdf (str): The file path to the SDF containing the docked 3D ligand poses.
        receptor_pdb (str): The file path to the PDB containing the target protein structure.
        
    Returns:
        Tuple[float, Optional[object]]: A tuple containing the fraction of poses passing ALL 
        checks (as a python float) and the full underlying pandas DataFrame for logging.
        
    Example:
        >>> valid_rate, detailed_df = pose_valid_rate("ligands_docked.sdf", "kras_g12c.pdb")
        >>> print(f"% PoseBusters-valid: {valid_rate * 100}%")
    """
    # Execute the core PoseBusters evaluation and retrieve the raw results DataFrame
    df = bust(docked_sdf, receptor_pdb)                                                     # Execute the underlying bust evaluation to retrieve the detailed per-pose boolean results DataFrame.
    # Check if every single column ("clash", "strain", "planarity", "bond", "angle") 
    # passed per row (all(axis=1)), and assign True if it did, False if it didn't. 
    # Then calculate the fraction of passed poses (.mean()) by averaging across all rows, 
    # cast the fraction to a python float, and return alongside the DataFrame.
    return float(df.all(axis=1).mean()), df                                                 