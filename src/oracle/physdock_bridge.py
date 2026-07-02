"""
=======================
PhysDock Oracle Bridge
=======================

This script acts as the crucial Stage-2 bridge to the existing PhysDock pipeline,
allowing the RL generator to run expensive physics ground-truth evaluations on its
top-k generated molecules.

How it works:
Because PhysDock's PyTorch dependencies conflict with the generator's environment,
this script intentionally does NOT import PhysDock directly. Instead, it assumes
PhysDock lives in its OWN isolated conda environment. It bridges the gap by writing
the candidate molecules to disk and calling the stage scripts via subprocesses using
`conda run -n <env> python <repo>/scripts/<stage>.py`.

PhysDock chain for the top-k molecules:
    1. DiffDock-L generates ligand poses.
    2. OpenMM relaxes/rescores only those poses.
    3. Boltz-2 is a seperate co-fold confidence signal (e.g. ipTM/pLDDT); not the
       structure that OpenMM relaxes. 

Since exact CLI flags and environment configurations vary, the script names and
argument templates are exposed as configurable parameters. It runs DiffDock-L, then
OpenMM on the DiffDock-L poses, then (optionally) Boltz-2 as an independent confidence
check, and finally parses a JSON summary into a consolidated result dictionary.
"""

import json
import os
import subprocess
from typing import Dict, List, Optional


def run_physics_on_topk(smiles: List[str], physdock_repo: str, conda_env: str = "physdock",
                        diffdock_script: str = "scripts/03_diffdock_l.py",
                        relax_script: str = "scripts/05_openmm_relax.py",
                        boltz_script: Optional[str] = "scripts/04_run_boltz.py",
                        workdir: str = "results/physics",
                        poses_dir: str = "diffdock_poses",
                        run_boltz: bool = True,
                        extra_diffdock_args: Optional[List[str]] = None,
                        extra_relax_args: Optional[List[str]] = None,
                        extra_boltz_args: Optional[List[str]] = None) -> Dict:
    """
    Orchestrates the PhysDock physics ground truth on candidate molecules.

    Pipeline (matches PhysDock's real chain):
        1. DiffDock-L (`diffdock_script`) generates ligand poses for the top-k molecules.
        2. OpenMM (`relax_script`) relaxes / rescores THOSE DiffDock-L poses (strain, energy).
        3. OPTIONAL, INDEPENDENT: Boltz-2 (`boltz_script`) co-folds for a confidence signal
           (ipTM/pLDDT). This is a separate check, not the input to OpenMM.

    Because PhysDock lives in its own conda env, every stage is launched with
    `conda run -n <env>`. Set the three *_script names to your actual PhysDock filenames.

    Args:
        smiles (List[str]): Top-k SMILES to evaluate.
        physdock_repo (str): Path to the root of the PhysDock repository.
        conda_env (str, optional): Isolated conda env for PhysDock. Defaults to "physdock".
        diffdock_script (str, optional): DiffDock-L pose-generation script (relative to repo).
        relax_script (str, optional): OpenMM relaxation script that consumes DiffDock-L poses.
        boltz_script (Optional[str], optional): Boltz-2 co-fold confidence script (independent).
        workdir (str, optional): Directory for inputs/outputs. Defaults to "results/physics".
        poses_dir (str, optional): Sub-dir (under workdir) where DiffDock-L writes poses and
            from which OpenMM reads them. Defaults to "diffdock_poses".
        run_boltz (bool, optional): Whether to run the independent Boltz-2 check. Defaults to True.
        extra_diffdock_args (Optional[List[str]], optional): Extra CLI flags for DiffDock-L.
        extra_relax_args (Optional[List[str]], optional): Extra CLI flags for OpenMM relax.
        extra_boltz_args (Optional[List[str]], optional): Extra CLI flags for Boltz-2.

    Returns:
        Dict: A dictionary containing the execution status ('ran'), the working directory ('workdir'), 
        return codes for the sub-scripts ('diffdock_rc', 'openmm_rc', 'boltz_rc'), any parsing notes 
        ('note'), and the parsed metrics payload ('summary') if successful.

    Example:
        >>> smiles_list = ["c1ccccc1", "CCO"]
        >>> res = run_physics_on_topk(smiles_list, physdock_repo="/path/to/PhysDock")
        >>> print(res["ran"])
        True
    """
    # -----------------------------------------------------------------------------------------
    # Workspace & Input Preparation
    # Create the output dir, validate the repo, and dump the SMILES to disk for PhysDock.
    # -----------------------------------------------------------------------------------------
    os.makedirs(workdir, exist_ok=True)                                                             # Create the destination working directory on disk, ignoring errors if it already exists
    if not physdock_repo or not os.path.isdir(physdock_repo):                                       # Validate that the provided PhysDock repository path points to an actual existing directory
        return {"ran": False, "note": f"PhysDock repo not found at {physdock_repo!r}.", "workdir": workdir}  # Abort execution and return a failure dictionary containing an error note if the repository is missing
    smi_file = os.path.join(workdir, "topk.smi")                                                    # Construct the absolute path for the temporary input file holding the candidate SMILES strings
    with open(smi_file, "w") as f:                                                                  # Open the newly created SMILES file in standard write mode to prepare for data dumping
        f.write("\n".join(smiles))                                                                  # Join the python list of SMILES strings with newlines and write them into the text file for PhysDock to ingest

    def _run(script, extra, flags=None):
        """
        Internal helper closure to execute a specific PhysDock script via the shell.
        
        Builds a `conda run` command string using the parent function's environment 
        variables. Executes the command synchronously via `subprocess.run`, enforcing 
        a 120-minute timeout to prevent infinitely hanging heavy physics simulations.
        
        Args:
            script (str): The specific script to run, relative to the PhysDock repository root.
            extra (List[str] or None): Additional command line arguments to append.
            flags (List[str] or None): Internal stage-specific command line arguments to append.
            
        Returns:
            subprocess.CompletedProcess: The result object containing return codes and stdout/stderr.
        """
        # -------------------------------------------------------------------------------------
        # Subprocess Execution Helper: Build the exact CLI string and invoke it synchronously 
        # with a strict timeout, to run as a subprocess in the isolated conda environment.
        # -------------------------------------------------------------------------------------
        cmd = ["conda", "run", "-n", conda_env, "python", os.path.join(physdock_repo, script),      # Construct the shell command list targeting the isolated conda environment and the specific python script
               "--smiles", smi_file, "--out", workdir] + (flags or []) + (extra or [])              # Append standard I/O flags, plus any stage-specific internal flags and user-provided extras
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60 * 120)                # Execute the assembled command synchronously, capturing stdout/stderr as text, and enforcing a strict 120-minute timeout

    # -----------------------------------------------------------------------------------------
    # Physics Pipeline Orchestration & Parsing: Run DiffDock-L on top-k SMILES, run OpenMM on 
    # DiffDock-L poses, optionally run Boltz-2 and parse summary of metrics.
    # -----------------------------------------------------------------------------------------
    try:                                                                                            # Wrap the heavy computational pipeline in a try-except block to gracefully catch timeouts or execution crashes
        # Invoke the helper to execute DiffDock-L, generating initial ligand poses for the top-k molecules
        dd = _run(diffdock_script, extra_diffdock_args)                                             
        # Invoke the helper to execute the OpenMM relaxation/rescoring script on the generated DiffDock-L poses
        rx = _run(relax_script, extra_relax_args,                                                   
                  flags=["--poses", os.path.join(workdir, poses_dir)])                              # Point the OpenMM relax stage specifically at the DiffDock-L pose output subdirectory
        bz_rc = None                                                                                # Initialize the Boltz-2 return code to None, remaining so unless the independent check is explicitly run
        # Check if the optional, independent Boltz-2 co-fold confidence evaluation is enabled
        if run_boltz and boltz_script:                                                              
            # Execute Boltz-2 and capture only its return code, as it serves as a separate confidence signal
            bz_rc = _run(boltz_script, extra_boltz_args).returncode                                 
        summary = {}                                                                                # Initialize an empty python dictionary to hold the parsed evaluation metrics if the pipeline succeeds
        # Check if the expected summary JSON file was successfully generated, 
        # and parse it into the local summary dictionary
        sp = os.path.join(workdir, "summary.json")                                                  # Construct the expected file path where PhysDock writes its final aggregated JSON results
        if os.path.exists(sp):                                                                      # Check if the expected summary JSON file was successfully generated by the external physics scripts
            summary = json.load(open(sp))                                                           # Open the JSON file and parse its contents directly into the local summary dictionary
        # Return a successful payload dictionary containing the target working directory,
        # the exit codes for each stage, and the fully parsed summary metrics
        return {"ran": True, "workdir": workdir,                                                    # Return a successful payload dictionary indicating execution completed with the target working directory
                "diffdock_rc": dd.returncode, "openmm_rc": rx.returncode, "boltz_rc": bz_rc,        # Inject the exit codes for DiffDock-L, OpenMM, and the optional Boltz-2 runs
                "summary": summary,                                                                 # Inject the fully parsed summary JSON dictionary containing strain, energy, affinity, and ipTM metrics
                "note": "DiffDock-L poses -> OpenMM relaxation done; Boltz-2 co-fold confidence separate."}  # Provide a descriptive note outlining the completed pipeline stages and expected output metrics
    except Exception as e:                                                                          # Catch any unexpected python errors, system crashes, or subprocess timeouts raised during execution
        return {"ran": False, "workdir": workdir, "note": f"PhysDock call failed: {e}"}             # Return a failure payload dictionary capturing the exact exception message for downstream debugging