"""
=======================
PhysDock Oracle Bridge
=======================

This script acts as the crucial Stage-2/3 bridge to the existing PhysDock pipeline,
allowing the RL generator to run expensive physics ground-truth evaluations on its
top-k generated molecules.

How it works:
Because PhysDock's PyTorch dependencies conflict with the generator's environment,
this script intentionally does NOT import PhysDock directly. Instead, it assumes
PhysDock lives in its own isolated conda environment. It bridges the gap by writing
the candidate molecules to disk and calling the stage scripts via subprocesses using
`conda run -n <env> python <repo>/scripts/<stage>.py`.

PhysDock chain for the top-k molecules:
    1. DiffDock-L generates ligand poses.
    2. OpenMM relaxes/rescores only those poses.
    3. Boltz-2 is a separate co-fold confidence signal (e.g. ipTM/pLDDT); not the
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
                        config_yaml: str = "configs/kras_g12c.yaml",
                        prepare_script: str = "scripts/01_prepare_target.py",
                        gate_script: str = "scripts/02_chem_gate.py",
                        diffdock_script: str = "scripts/03_run_diffdock.py",
                        relax_script: str = "scripts/05_physics_rescore.py",
                        boltz_script: Optional[str] = "scripts/04_run_boltz.py",
                        ensemble_script: str = "scripts/06_ensemble_analysis.py",
                        report_script: str = "scripts/07_evaluate_and_report.py",
                        workdir: str = "results/physics",
                        poses_dir: str = "diffdock_poses",
                        run_boltz: bool = True,
                        max_ligands: int = 4,
                        top_per_ligand: int = 3,
                        timeout_min: int = 120,
                        extra_diffdock_args: Optional[List[str]] = None,
                        extra_relax_args: Optional[List[str]] = None,
                        extra_boltz_args: Optional[List[str]] = None) -> Dict:
    """
    Orchestrates the PhysDock physics ground truth on the top-k candidate molecules.

    Pipeline (matches PhysDock's real chain):
        1. DiffDock-L (`diffdock_script`) generates ligand poses for the top-k molecules.
        2. OpenMM (`relax_script`) relaxes / rescores the DiffDock-L poses (strain, energy).
        3. Optional, Independent: Boltz-2 (`boltz_script`) co-folds for a confidence signal
           (ipTM/pLDDT). This is a separate check, not the input to OpenMM.

    Because PhysDock lives in its own conda env, every stage is launched with
    `conda run -n <env>`. Set the three *_script names to your actual PhysDock filenames.

    Args:
        smiles (List[str]): Top-k SMILES to evaluate.
        physdock_repo (str): Path to the root of the PhysDock repository.
        conda_env (str, optional): Isolated conda env for PhysDock. Defaults to "physdock".
        config_yaml (str, optional): Target config filename. Defaults to "configs/kras_g12c.yaml".
        prepare_script (str, optional): Target prep script path. Defaults to "scripts/01_prepare_target.py".
        gate_script (str, optional): Chemical filtering script path. Defaults to "scripts/02_chem_gate.py".
        diffdock_script (str, optional): DiffDock-L pose-generation script (relative to repo).
        relax_script (str, optional): OpenMM relaxation script that consumes DiffDock-L poses.
        boltz_script (Optional[str], optional): Boltz-2 co-fold confidence script (independent).
        ensemble_script (str, optional): Conformational spread analysis script. Defaults to "scripts/06_ensemble_analysis.py".
        report_script (str, optional): Reporting and parsing script. Defaults to "scripts/07_evaluate_and_report.py".
        workdir (str, optional): Directory for inputs/outputs. Defaults to "results/physics".
        poses_dir (str, optional): Sub-dir (under workdir) where DiffDock-L writes poses and from which OpenMM reads them. Defaults to "diffdock_poses".
        run_boltz (bool, optional): Whether to run the independent Boltz-2 check. Defaults to True.
        max_ligands (int, optional): AWS protection budget parameter capping Boltz ligands. Defaults to 4.
        top_per_ligand (int, optional): OpenMM relaxation budget filter cap. Defaults to 3.
        timeout_min (int, optional): Default timeout limit for heavy GPU blocks. Defaults to 120.
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
    # --------------------------------------------------------------------------------------------------------
    # Workspace & Repo Validation
    # PhysDock is a self-contained, CONFIG-DRIVEN pipeline: its scripts read a target YAML and a
    # prepared-target JSON, not a loose --smiles/--out pair. So rather than invent a CLI PhysDock
    # does not expose, this bridge injects the RL's generated candidates into PhysDock's own target_prep.json
    # and then runs its real staged scripts (03 DiffDock-L -> 04 Boltz-2 -> 05 OpenMM -> 06/07 report),
    # honouring the budget caps PhysDock already bakes in (--max-ligands, --top-per-ligand).
    # --------------------------------------------------------------------------------------------------------
    
    # Create the working directory for the bridge, and validate the PhysDock repo path exists and is valid.
    os.makedirs(workdir, exist_ok=True)                                                             # Create the bridge working directory (used only for our own notes/manifests)
    if not physdock_repo or not os.path.isdir(physdock_repo):                                       # Validate that the provided PhysDock repository path points to an actual existing directory
        return {"ran": False, "note": f"PhysDock repo not found at {physdock_repo!r}.", "workdir": workdir}  # Abort with a failure note if the repo is missing

    # PhysDock's canonical prepared-target registry that every GPU stage reads from
    prep_path = os.path.join(physdock_repo, "data/processed/target_prep.json")                      

    def _run(script, flags=None, timeout_min=120):
        """
        Execute one PhysDock stage script inside its isolated conda env, from the repo root.

        PhysDock scripts are invoked as `python scripts/NN_*.py --config <cfg> [caps...]` with the
        repository as the working directory (they use repo-relative paths internally). This helper
        wraps that call in `conda run -n <env>` so PhysDock's heavyweight, dependency-conflicting
        stack (DiffDock-L / Boltz-2 / OpenMM) stays fully isolated from the RL generator's env.

        Args:
            script (str): Stage script path relative to the PhysDock repo (e.g. "scripts/03_run_diffdock.py").
            flags (List[str] or None): Stage-specific CLI flags to append after `--config`.
            timeout_min (int): Hard wall-clock cap in minutes for this stage.

        Returns:
            subprocess.CompletedProcess: The completed process (returncode + captured stdout/stderr).
        """
        cmd = ["conda", "run", "-n", conda_env, "python", script, "--config", config_yaml] + (flags or [])          # Build the config-driven stage command PhysDock actually expects
        return subprocess.run(cmd, cwd=physdock_repo, capture_output=True, text=True, timeout=60 * timeout_min)     # Run from the repo root so PhysDock's relative paths resolve, with a strict timeout

    # ---------------------------------------------------------------------------------------------------
    # PhysDock's Stage 01 (once): If not already done, prepare the reference receptor + crystal controls.
    # This provides a real KRAS G12C receptor (6OIM) and pocket center to dock the survivors into.
    # ---------------------------------------------------------------------------------------------------
    if not os.path.exists(prep_path):                                                               # Only pay the one-time preparation cost if the registry does not yet exist
        # Run Stage-01 target preparation (downloads/cleans crystal structures, extracts controls)
        pr = _run(prepare_script, timeout_min=30)                                                   
        if not os.path.exists(prep_path):                                                           # If preparation still produced no registry, we cannot proceed
            return {"ran": False, "workdir": workdir,                                               # Surface the failure with the captured stderr for debugging
                    "note": f"Stage-01 target prep failed:\n{getattr(pr, 'stderr', '')[-2000:]}"}

    # ------------------------------------------------------------------------------------------------
    # Inject the RL agent's generated molecules (the "survivors") directly into PhysDock's internal 
    # configuration registry (target_prep.json) pointing at the reference (6OIM) receptor: 
    # Because the RL agent generates 2D SMILES strings without 3D coordinates, this step securely
    # "borrows" the 3D protein (6OIM sotorasib) receptor and pocket coordinates from a known 
    # ligand-receptor crystal structure so that downstream physics engines know exactly where to dock
    # the new molecules.
    # 
    # DiffDock-L docks blind (no box needed), so generated SMILES with no crystal structure are
    # perfectly valid inputs — we simply reuse the prepared reference receptor + pocket center.
    # ------------------------------------------------------------------------------------------------
    
    # Load PhysDock's prepared-target registry and 
    # grab the existing ligand dict (crystal controls from Stage-01).
    prep = json.load(open(prep_path))                                                               # Load PhysDock's prepared-target registry
    ligs = prep.get("ligands", {})                                                                  # Grab the existing ligand dict (crystal controls from Stage-01)
    ref = None                                                                                      # Will hold the reference ligand entry whose receptor/pocket we borrow for the survivors
    # Iterate through the existing ligands to find the 6OIM reference (covalent sotorasib co-crystal)
    # and adopt it as the receptor/pocket donor for the generated molecules. 
    for _lid, _info in ligs.items():                                                                # Prefer the 6OIM covalent reference (matches the Cys12 warhead chemistry)
        if str(_info.get("pdb_id", "")).upper().startswith("6OIM"):                                 # Match the sotorasib co-crystal used as the covalent docking reference
            ref = _info; break                                                                      # Adopt it and stop searching
    # If the 6OIM reference is not found, fall back to the first prepared ligand in the registry.
    if ref is None and ligs:                                                                        # Fall back to any prepared ligand if 6OIM is somehow absent
        ref = next(iter(ligs.values()))                                                             # Use the first available prepared entry as the receptor/pocket donor
    # If no prepared ligand exists at all, abort and raise an error.
    if ref is None:                                                                                 # If there is no prepared ligand at all, we have no receptor to dock into
        return {"ran": False, "workdir": workdir, "note": "No prepared reference receptor in target_prep.json."}
    
    # For each generated candidate SMILES string, create a new unique ligand entry in the registry 
    # (formatted sequentially as gen_0000, gen_0001, etc.) pointing at the reference receptor/pocket. 
    # It copies the reference receptor SDF file and pocket center coordinates from the reference into the 
    # new candidate ligand's entry, marks the provenance with tags such as "rl_generated", and nulls out 
    # the "ref_sdf" and "pchembl" values since no experimental data exists for the generated molecules.
    for i, smi in enumerate(smiles):                                                                # Register each RL survivor under a stable generated id
        ligs[f"gen_{i:04d}"] = {"pdb_id": ref.get("pdb_id"), "receptor_pdb": ref.get("receptor_pdb"),  # Point the generated ligand at the reference receptor structure
                                "ref_sdf": None, "smiles": smi, "pocket_center": ref.get("pocket_center"),  # Carry the pocket center; no crystal reference pose exists for a generated molecule
                                "pchembl": None, "role": "rl_generated"}                            # Mark provenance so the report can separate generated survivors from crystal controls
    # Write the augmented ligand set back into the registry object and persist it to disk 
    # so every downstream stage (e.g., DiffDock-L, Boltz-2) sees the survivors.
    prep["ligands"] = ligs                                                                          # Write the augmented ligand set back into the registry object
    json.dump(prep, open(prep_path, "w"), indent=2)                                                 # Persist the injected registry so every downstream stage sees the survivors

    # ---------------------------------------------------------------------------------------------
    # Staged physics: re-gate (so gen ids are cleared) -> DiffDock-L -> Boltz-2 (capped) ->
    # OpenMM rescore (capped) -> ensemble -> report. Each stage honours PhysDock's own budget caps.
    # ---------------------------------------------------------------------------------------------
    try:                                                                                            # Wrap the heavy multi-stage physics run so any timeout/crash returns a clean failure payload
        rcs = {}                                                                                    # Collect per-stage return codes for the caller to inspect
        # Run the chemical gate to clear the freshly injected gen_* ids for the GPU stages
        rcs["chem_gate"] = _run(gate_script, timeout_min=15).returncode                             
        # Run the DiffDock-L stage to generate poses for every cleared ligand (crystal controls + survivors)
        rcs["diffdock"] = _run(diffdock_script, extra_diffdock_args, timeout_min=timeout_min).returncode  
        bz_rc = None                                                                                # Boltz-2 return code stays None unless the independent co-fold check runs
        # If requested, run the independent Boltz-2 co-fold confidence stage (capped to max_ligands)
        if run_boltz and boltz_script:                                                                  # Optionally run the independent Boltz-2 confidence signal
            bz_rc = _run(boltz_script, ["--max-ligands", str(max_ligands)] + (extra_boltz_args or []),  # Cap Boltz-2 ligand count to protect the AWS budget (it is the most expensive stage)
                         timeout_min=timeout_min).returncode
        rcs["boltz"] = bz_rc                                                                        # Record the Boltz-2 exit code (or None if skipped)
        # Run the OpenMM relaxation/rescoring stage on the DiffDock-L poses (capped to top_per_ligand)
        rcs["physics"] = _run(relax_script, ["--top-per-ligand", str(top_per_ligand)] + (extra_relax_args or []),  # Stage-05 OpenMM: relax/rescore only the top-N DiffDock-L poses per ligand
                              timeout_min=timeout_min).returncode
        # Run the ensemble analysis stage for the top-N DiffDock-L poses (capped to max_ligands)
        rcs["ensemble"] = _run(ensemble_script, timeout_min=20).returncode                          # Stage-06: conformational spread/clustering analysis of the pose ensemble
        # Run the final report stage to aggregate all metrics into a Markdown/JSON report under results/report/
        rcs["report"] = _run(report_script, timeout_min=20).returncode                              # Stage-07: aggregate everything into the Markdown/JSON report

        # ------------------------------------------------------------------------------------
        # Parse the aggregated report/ensemble JSON that PhysDock writes into results/.
        # ------------------------------------------------------------------------------------
        
        # Probe the known PhysDock summary locations in priority order, and load the first one that exists.
        summary = {}                                                                                # Initialise the parsed-metrics payload
        for cand in ("results/report/summary.json", "results/report/report.json",                   # Probe the known PhysDock summary locations in priority order
                     "results/ensemble/ensemble.json", "results/physics/physics.json"):
            sp = os.path.join(physdock_repo, cand)                                                  # Resolve each candidate summary path against the repo root
            if os.path.exists(sp):                                                                  # Use the first summary file that actually exists
                try:                                                                                # Defensively parse so a malformed file cannot crash the bridge
                    summary = json.load(open(sp)); summary["_source"] = cand; break                 # Load it, tag its provenance, and stop probing
                except Exception:                                                                   # Ignore parse errors and keep probing the remaining candidates
                    continue
        # If no summary was found, return a failure payload with the staged return codes for debugging
        return {"ran": True, "workdir": workdir, "physdock_repo": physdock_repo,                    # Success payload: staged return codes plus the parsed summary
                "returncodes": rcs, "summary": summary,
                "note": "Survivors injected into target_prep.json; DiffDock-L -> Boltz-2 (capped) "
                        "-> OpenMM (capped) -> report. Report under results/report/."}
    
    # Catch any unexpected error, subprocess timeout, or OS failure raised during the physics run
    except Exception as e:                                                                          
        return {"ran": False, "workdir": workdir, "note": f"PhysDock call failed: {e}"}             # Return a failure payload capturing the exact exception for downstream debugging