"""
=============================================
Stage-3 Mechanistic Pose Validation (Oracle)
=============================================

This script performs protein-ligand interaction profiling (fingerprinting) to validate 
the mechanistic quality of a docked molecule. Considering a raw docking score without 
interaction context as a weak evidence, this module computes which specific pocket residues
a docked pose actually contacts (using ProLIF, or PLIP as a conceptual alternative).

Specifically for the KRAS G12C target, it strictly requires engagement of the switch-II 
selectivity residues (His95, Tyr96, Gln99) and, if evaluating covalent poses, the Cys12 
anchor. If a pose scores highly but completely misses the selectivity groove, it is 
discarded. This converts an abstract numerical score into a mechanistic claim: 
"binds where sotorasib binds, through the interactions that confer G12C selectivity."

Resilience Strategy:
ProLIF computations run on the CPU. If the library is not installed or structural parsing 
fails, `engaged_switch_ii` returns `None`. The overarching RL pipeline treats this as 
an unavailable filter and "fails open" (bypasses the filter with a warning) rather than 
crashing the active learning loop.
"""

from typing import Optional, Set

# -----------------------------------------------------------------------------------------
# Target-Specific Constants
# Defines the critical mechanistic residue numbers required to classify a hit as active 
# against the KRAS G12C switch-II pocket.
# -----------------------------------------------------------------------------------------
SWITCH_II = {95, 96, 99}                                                                    # Define the core KRAS switch-II selectivity residues (His95, Tyr96, Gln99) as a set
CYS12 = 12                                                                                  # Define the integer residue number for the KRAS G12C covalent attachment point


def contacts(pose_sdf: str, receptor_pdb: str) -> Optional[Set[int]]:
    """
    Computes the set of receptor residue numbers that the ligand pose physically contacts.
    
    Attempts to import the ProLIF and MDAnalysis libraries. If successful, it parses the 
    receptor PDB and the ligand SDF, computes the Protein-Ligand Interaction Fingerprint (PLIF), 
    and iterates through the resulting Pandas DataFrame columns. It parses the multi-index 
    column labels (e.g., 'HIS95.A') to extract and aggregate the integer residue numbers.
    
    Args:
        pose_sdf (str): Filepath to the docked ligand pose in SDF format.
        receptor_pdb (str): Filepath to the target protein structure in PDB format.
        
    Returns:
        Optional[Set[int]]: A set containing the integer residue numbers the ligand interacts 
        with. Returns None if required libraries are missing or if 3D parsing fails.
        
    Example:
        >>> res_set = contacts("ligand_pose.sdf", "kras_g12c.pdb")
        >>> print(res_set)
        {12, 68, 95, 96, 99}
    """
    # -------------------------------------------------------------------------------------
    # Optional Dependency Loading
    # Attempt to import heavy bioinformatics libraries, failing gracefully if absent.
    # -------------------------------------------------------------------------------------
    try:                                                                                    # Wrap dependency imports in a try block to handle environments without them
        import prolif as plf                                                                # Import the ProLIF library for protein-ligand interaction fingerprinting
        import MDAnalysis as mda                                                            # Import MDAnalysis to parse and handle 3D coordinate trajectories and PDB files
    except Exception:                                                                       # Catch ImportErrors or related exceptions if the optional libraries are missing
        return None                                                                         # Return None safely, signaling to the caller that interaction filtering is unavailable
    
    # -------------------------------------------------------------------------------------
    # Fingerprint Calculation & Parsing: Load 3D coordinates of receptor and ligand, run the
    # ProLIF fingerprint, and extract residue integers.
    # -------------------------------------------------------------------------------------
    try:                                                                                    # Wrap the core computational interaction logic to catch file reading or parsing errors
        # Load the receptor PDB structure and (best) ligand SDF file,
        # and convert them to a ProLIF Molecule
        u_prot = mda.Universe(receptor_pdb)                                                 # Load the target protein PDB structure into an MDAnalysis Universe object
        prot = plf.Molecule.from_mda(u_prot)                                                # Convert the MDAnalysis Universe protein representation into a ProLIF Molecule object
        lig = plf.sdf_supplier(pose_sdf)[0]                                                 # Read the docked ligand pose from the SDF file and extract the first (best) molecule
        # Compute the ProLIF fingerprint between them, and extract 
        # the resulting binary interactions to a pandas DataFrame
        fp = plf.Fingerprint()                                                              # Initialize an empty ProLIF interaction Fingerprint generator
        fp.run_from_iterable([lig], prot)                                                   # Execute the fingerprint calculation between the ligand pose and the protein
        df = fp.to_dataframe()                                                              # Extract the resulting binary interactions into a pandas DataFrame
        resids = set()                                                                      # Initialize an empty set to accumulate unique integer residue numbers contacted
        
        # Iterate over the DataFrame columns, corresponding to specific 
        # formed interactions and parse the multi-index column labels 
        # (e.g., ('LIG1', 'HIS95.A', 'Hydrophobic')) to extract and 
        # aggregate the integer residue numbers.
        for col in df.columns:                                                              # Iterate over the DataFrame columns, which correspond to specific formed interactions
            for level in col:                                                               # Iterate through the MultiIndex levels containing protein residue labels
                s = str(level)                                                              # Cast the current MultiIndex level to a string for substring parsing
                # Isolate the numeric portion of the residue label 
                # (e.g., '95' from 'HIS95.A') and convert to integer
                num = "".join(ch for ch in s if ch.isdigit())                               
                if num:                                                                     # Check if the extraction yielded a valid numeric string
                    resids.add(int(num)); break                                             # Cast to integer, add to the contacted residues set, and break to the next column
        # Return the completely populated set of contacted protein residue integers
        return resids                                                                       
    except Exception:                                                                       # Catch any errors occurring during 3D parsing, fingerprinting, or dataframe iteration
        return None                                                                         # Yield None to safely fail open if the interaction analysis crashes internally


def engaged_switch_ii(pose_sdf: str, receptor_pdb: str, need_cys12: bool = False) -> Optional[bool]:
    """
    Validates whether a docked pose engages the essential KRAS G12C switch-II residues.
    
    Calls `contacts()` to get the set of all interacting residues. It then intersects this 
    set with the predefined `SWITCH_II` set (His95, Tyr96, Gln99). A pose is deemed valid 
    if it contacts at least two of these switch-II residues. If `need_cys12` is True, it 
    also strictly requires an interaction with residue 12.
    
    Args:
        pose_sdf (str): Filepath to the docked ligand pose.
        receptor_pdb (str): Filepath to the target receptor structure.
        need_cys12 (bool, optional): Whether to strictly require a contact at Cys12. Defaults to False.
        
    Returns:
        Optional[bool]: True if the pose meets mechanistic criteria, False if it misses 
        the crucial residues, or None if the interaction computation was unavailable.
        
    Example:
        >>> is_mechanistic = engaged_switch_ii("pose.sdf", "protein.pdb", need_cys12=True)
        >>> print(is_mechanistic)
        True
    """
    # ---------------------------------------------------------------------------------------
    # Mechanistic Logic Resolution: Fetch a set of all the interacting residues ("contacts"),
    # and execute set mathematics to prove structural engagement.
    # ---------------------------------------------------------------------------------------
    c = contacts(pose_sdf, receptor_pdb)                                                    # Compute the complete set of contacted receptor residues via the helper function
    if c is None:                                                                           # Check if the interaction profiling failed or if dependencies were entirely missing
        return None                                                                         # Propagate the None signal upward so the overarching pipeline can fail open
    # Evaluate true if the intersection of contacted residues and Switch-II targets is >= 2
    ok = len(SWITCH_II & c) >= 2                                                            
    # If the caller explicitly requires covalent engagement with the Cys12 residue, 
    # intersect the current validity flag with the strict presence of Cys12 in the contacts
    if need_cys12:                                                                          # Check if the caller explicitly requires covalent engagement with the Cys12 residue
        ok = ok and (CYS12 in c)                                                            # Intersect the current validity flag with the strict presence of Cys12 in the contacts
    # Return the final boolean flag indicating if the specific mechanistic pose is valid
    return ok                                                                               