"""
======================================
Active-Learning Orchestrator (Stage 3)
======================================

This script acts as the closed-loop active-learning orchestrator for Stage 3 of the pipeline.
It bridges the deep generative policy (the GNN agent) with the computationally expensive 
physics-based oracle (Gnina docking) using a fast surrogate model as a proxy.

It runs R rounds of the proxy <-> oracle cycle:
  1. GENERATE   sample a large pool of molecules from the current policy (PPO agent; GFlowNet handled by its own run)
  2. ACQUIRE    surrogate-score the pool, pick an information-rich, chemically diverse batch using UCB (acquire.py)
  3. ORACLE     covalently dock the batch to Cys12 (gnina-GPU) + PoseBusters + interaction filter
  4. LABEL      append (SMILES, calibrated oracle score) to the surrogate training CSV
  5. RETRAIN    retrain the deep ensemble on the augmented set (subprocess -> train_surrogate.py)
  6. MEASURE    log scaffold-split Spearman, calibration, uncertainty-on-generated, and top-k oracle score

The four measured curves (written to metrics_round_*.json) act as the empirical proof that 
the loop works: they should improve monotonically across rounds. See eval/al_curves.py to plot them.

Architectural note: This orchestrator deliberately shells out to `train_surrogate.py` for 
retraining. This reuses the proven, tested training code and completely isolates PyTorch 
memory contexts to prevent creeping memory leaks across long active-learning rounds.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from surrogate.predict import AffinityScorer                      
from activelearn.acquire import select_batch                      
from rdkit import RDLogger                                        
RDLogger.DisableLog("rdApp.*")                                    
from oracle.dock import dock_many, dock_many_fast, which_engine   
from activelearn.calibrate_dock import dock_score_of, apply_calibration, calibrate  
from oracle.covalent_dock import covalent_dock_many               
from oracle.prepare_receptor import ligand_com                    
from oracle.cache import DockCache                                
from reward.warhead import has_warhead                            


def _canon(s):
    """
    Standardizes a given SMILES string into its canonical representation.
    
    Translates the raw string into an RDKit Mol object, then converts it back to 
    a SMILES string. This normalizes different string representations of the same 
    molecule, ensuring safe sets and deduplication.
    
    Args:
        s (str): The raw input SMILES string.
        
    Returns:
        str or None: The canonicalized SMILES string, or None if parsing fails.
    """
    # -----------------------------------------------------------------------------------------
    # Canonicalization
    # -----------------------------------------------------------------------------------------
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None                       # Parse string to Mol, return standardized string if valid, else safely return None


def dock_batch(smiles, cfg, receptor, center, size, pose_dir, cache=None):
    """
    Docks an acquired batch and returns pKd-scaled scores with their docking mode.

    Warhead-bearers are covalently docked to Cys12 (gnina-only); everything else is docked
    non-covalently. Both paths return a score on the pKd scale (gnina CNNaffinity verbatim, or
    -dG/1.364 for a Vina-family engine), NOT a raw kcal/mol affinity -- so the value is the same
    physical quantity as the ChEMBL pChEMBL labels it will be merged with, and a frozen calibration
    can map it onto the training scale without any per-round renormalisation.

    gnina's CNNaffinity is preferred deliberately: it is the signal that was not fooled by the
    lipophilic macrocycles that exploited the Vina score in earlier runs.

    A shared DockCache makes any molecule already docked under these parameters return instantly, at
    zero cost to accuracy since every dock is seed-locked.

    Args:
        smiles (List[str]): The acquired batch.
        cfg (dict): Parsed target config containing docking parameters.
        receptor (str): Prepared receptor path.
        center: Docking box centre coordinates (x, y, z).
        size: Docking box dimensions in Angstroms.
        pose_dir (str): Directory for covalent pose SDFs.
        cache (DockCache or None): Shared, seed-locked docking cache to accelerate repeated docking.

    Returns:
        List[Tuple[str, float, str]]: A list of tuples containing (smiles, pKd_scaled_score, mode) 
        where mode is in {"covalent", "noncovalent"}.
    """
    # ----------------------------------------------------------------------------------------------
    # Warhead Detection & Routing: Separate the batch based on reactive capability (covalent vs non)
    # for proper physics simulation, and extract the covalent target specification from the config.
    # ----------------------------------------------------------------------------------------------
    cov = [s for s in smiles if has_warhead(Chem.MolFromSmiles(s))]                          # Filter SMILES containing a defined reactive covalent warhead motif, routing them to Cys12 docking
    noncov = [s for s in smiles if s not in set(cov)]                                        # Collect all remaining SMILES to be processed as standard non-covalent ligands
    rows = []                                                                                # Accumulate (smiles, score, mode) triples for the unified scoring results
    cys = cfg.get("covalent", {}).get("cys_spec", "A:12:SG")                                 # Extract covalent target specification (defaulting to KRAS Cys12 sulfur)
    receptor_cov = cfg.get("receptor_pdb") or receptor                                       # COVALENT MUST USE THE PDB: --covalent_rec_atom needs chain/residue naming that PDBQT strips
    cnn_mode = cfg["docking"].get("cnn_scoring", "rescore")                                  # Extract gnina CNN scoring mode from configuration
    engine = which_engine(cfg["docking"].get("engine", "gnina")) or "smina"                  # Fallback logic to detect the engine actually present on this machine
    
    # ------------------------------------------------------------------------------------------------------
    # Covalent Docking Execution: If the covalent pool is non-empty, route to the covalent docking routine.
    # Covalent docking is gnina-only; skip cleanly when gnina is unavailable.
    # ------------------------------------------------------------------------------------------------------
    if cov and which_engine("gnina") == "gnina":                                             # Verify covalent candidates exist and that gnina is successfully installed in the PATH
        for r in covalent_dock_many(cov, receptor_cov, cys, autobox_ligand=cfg["ref_ligand_sdf"], engine="gnina", cnn_scoring=cnn_mode,
                                    keep_pose_dir=pose_dir, cache=cache):                    # Dock every warhead-bearer to Cys12, keeping poses for downstream validation
            v = dock_score_of(r, "gnina")                                                    # Extract pKd-scaled score (CNNaffinity) using the new calibration extractor
            if v is not None:                                                                # Validate that the docking completed without geometric or systemic failure
                rows.append((r.smiles, v, "covalent"))                                       # Tag the mode as 'covalent' so the frozen per-mode calibration is applied correctly later
                
    # -------------------------------------------------------------------------------------------------------------
    # Non-Covalent Docking Execution: Dispatch the non-reactive pool to standard localized box docking.
    # -------------------------------------------------------------------------------------------------------------
    for r in dock_many_fast(noncov, receptor, center, size, engine=engine, cnn_scoring=cnn_mode,
                            gpu=cfg["docking"].get("gpu", True), progress=True, cache=cache): # Batch-dock the non-covalent remainder using config-driven engine and GPU flags
        v = dock_score_of(r, engine)                                                         # Extract pKd-scaled score (CNNaffinity for gnina, -dG/1.364 for smina)
        if v is not None:                                                                    # Validate successful structural calculation
            rows.append((r.smiles, v, "noncovalent"))                                        # Tag the mode as 'noncovalent' and append to results
    return rows                                                                              # Return the fully compiled and aligned list of (smiles, pKd_score, mode)


def _carve_fixed_holdout(base_csv, out_dir, frac=0.15, seed=0):
    """
    Freezes one scaffold-based holdout ONCE, kept out of every round's training.
    
    Extracts the Murcko scaffolds for all molecules in the base dataset, groups them by 
    scaffold, and randomly selects entire scaffold groups until the requested fraction is met. 
    This creates a structurally distinct holdout set that is evaluated each round for a 
    comparable Spearman correlation. By keeping this set stable, we isolate real distribution 
    shift measurements from moving-split noise.
    
    Args:
        base_csv (str): Path to the initial surrogate training dataset.
        out_dir (str): Output directory to write the fixed holdout CSV.
        frac (float): Fraction of data to reserve for the holdout set. Defaults to 0.15.
        seed (int): Random seed for reproducible splitting. Defaults to 0.
        
    Returns:
        tuple: (train_df, holdout_csv_path) containing the reduced training dataframe and 
        the path to the newly written static holdout file.
    """
    # -----------------------------------------------------------------------------------------
    # Semantic Group: Fixed Scaffold Split
    # Group molecules by their Bemis-Murcko core to ensure strict structural division.
    # -----------------------------------------------------------------------------------------
    df = pd.read_csv(base_csv); smi_col = df.columns[0]; groups = {}                                # Load base CSV, identify the SMILES column, and initialize the scaffold grouping dictionary
    for i, smi in enumerate(df[smi_col].astype(str)):                                               # Iterate through every SMILES string in the dataset with its index
        try:                                                                                        # Wrap scaffold extraction to handle invalid or complex molecules
            from rdkit.Chem.Scaffolds import MurckoScaffold                                         # Import MurckoScaffold dynamically
            m = Chem.MolFromSmiles(smi)                                                             # Parse the SMILES string into an RDKit Mol object
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ""               # Extract the generalized core scaffold SMILES, reverting to empty string if missing
        except Exception:                                                                           # Catch any structural parsing errors
            key = ""                                                                                # Group unparseable molecules under the empty string key
        groups.setdefault(key, []).append(i)                                                        # Append the row index to the corresponding scaffold group list
        
    keys = sorted(groups.keys()); np.random.RandomState(seed).shuffle(keys)                         # Sort scaffold keys deterministically, then shuffle them using the fixed random seed
    target = int(len(df) * frac); hold = []                                                         # Calculate the target number of holdout molecules and initialize the holdout index list
    for k in keys:                                                                                  # Iterate through the shuffled scaffold groups
        if len(hold) >= target: break                                                               # Halt accumulation once the requested holdout fraction is reached or exceeded
        hold.extend(groups[k])                                                                      # Add all indices from the current scaffold group into the holdout list
        
    hset = set(hold)                                                                                # Convert the holdout list to a set for O(1) membership checking
    hold_df = df.iloc[sorted(hset)]                                                                 # Extract the reserved holdout rows into a distinct dataframe
    train_df = df.iloc[[i for i in range(len(df)) if i not in hset]]                                # Extract the remaining rows to form the pruned training dataframe
    hpath = os.path.join(out_dir, "fixed_holdout.csv"); hold_df.to_csv(hpath, index=False)          # Save the frozen holdout dataframe to disk to persist across all rounds
    return train_df, hpath                                                                          # Return the workable training dataframe and the path to the static holdout


def run(agent_ckpt, make_env, node_dim, num_actions, config, base_csv, out_dir,
        rounds=3, pool=3000, n_dock=250, device="cuda", surrogate_dir="../artifacts/surrogate_kras",
        physdock_repo=None, hidden_dim=128, edge_dim=4, pocket_npy=None):
    """
    Executes the multi-round active learning loop connecting generation, scoring, and retraining.
    
    Loads the trained PyG actor-critic agent and restores its ESM pocket embedding condition. 
    It then loops through predefined `rounds`. In each round, it generates thousands of candidate 
    SMILES, uses the surrogate ensemble to select a diverse high-confidence subset (via UCB), 
    evaluates that subset using actual Gnina docking, and applies a frozen calibration mapping 
    to append physics-backed labels to the training set. It then shells out to train a fresh deep 
    ensemble surrogate, records performance metrics (against a static holdout), and repeats.
    
    Robustness features include a resume guard (skips completed rounds if execution is interrupted)
    and a disk-backed JSONL docking cache (eliminates redundant physics calculations).
    """
    # -----------------------------------------------------------------------------------------
    # Initialization & Configuration Parsing
    # Prepare directories, load physics targets, and setup cross-session caching.
    # -----------------------------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)                                                             # Ensure the root active learning output directory exists
    cfg = yaml.safe_load(open(config))                                                              # Parse the master YAML configuration file defining the docking grid
    receptor = cfg.get("receptor_pdbqt") or cfg["receptor_pdb"]                                     # Extract target receptor file path, prioritizing optimized PDBQT
    if not os.path.exists(receptor):                                                                # Check if the preferred receptor format actually exists on disk
        receptor = cfg["receptor_pdb"]                                                              # Fallback to standard PDB format if PDBQT is missing
    box = cfg["docking_box"]; center = box["center"] or ligand_com(cfg["ref_ligand_sdf"])           # Resolve 3D grid center: explicit config coords OR computed ligand center of mass
    size = tuple(box["size_angstrom"])                                                              # Cast the spatial box dimension constraints into an immutable tuple

    # Stage-3 acceleration + resumability: a single disk-backed docking cache lives inside out_dir.
    # It makes re-encountered molecules free (across rounds and across a resumed sessions), and 
    # because it is just a JSONL file it travels with the rest of the resume state.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")                                        # Silence HF tokenizer fork warnings when ESM is imported downstream
    cache = DockCache(os.path.join(out_dir, "dock_cache.jsonl"))                                    # Open (or resume) the persistent docking cache for this run

    # lazy imports of repo agent + sampler (kept here so this file imports without torch for testing)
    import torch                                                                                    # Import PyTorch locally to avoid heavy memory allocation during CLI --help checks
    from gnn_agent import MoleculeAgent                                                             # Import the architectural blueprint for the structural generator
    from activelearn.generate import sample_smiles                                                  # Import the inference routine for batch SMILES generation

    train_csv = os.path.join(out_dir, "surrogate_train_round0.csv")                                 # Define the filename for the initial round 0 starting dataset
    _train_df, _holdout_csv = _carve_fixed_holdout(base_csv, out_dir, frac=0.15, seed=0)            # Perform the static scaffold split to isolate the 15% evaluation holdout
    _train_df.to_csv(train_csv, index=False)                                                        # Write the remaining 85% base training data to the active learning working directory
    print(f"[stage3] fixed scaffold holdout: {len(open(_holdout_csv).readlines())-1} mols held out "
          f"as a stable per-round eval set -> {_holdout_csv}")                                      # Log the initialization of the fixed holdout for distribution shift tracking
    curr_surrogate = surrogate_dir                                                                  # Initialize pointer to track the most recently trained ensemble directory

    # -----------------------------------------------------------------------------------------
    # Semantic Group: Frozen Docking Calibration
    # Fit once (or reuse) a round-invariant map from docking score to the training label scale. 
    # This preserves absolute affinity information and makes rounds comparable, solving the 
    # information loss caused by the old per-round z-normalisation.
    # -----------------------------------------------------------------------------------------
    _cal_path = os.path.join(out_dir, "dock_calibration.json")                                      # Define path for the persistent docking-to-pChEMBL calibration mapping
    if os.path.exists(_cal_path):                                                                   # Check if the calibration map was already computed in a previous session
        _cal = json.load(open(_cal_path))                                                           # Reload the frozen calibration parameters from disk
        print(f"[stage3] reusing frozen dock calibration -> {_cal_path}")                           # Announce the reuse of the existing scale mapping
    else:                                                                                           # If no calibration exists, trigger the expensive one-off fitting process
        print("[stage3] fitting frozen dock->label calibration (one-off; cached thereafter)")       # Announce the start of the reference molecule docking required for mapping
        _cal = calibrate(base_csv, cfg, _cal_path, n_ref=200, seed=0, cache=cache)                  # Dock 200 reference molecules and fit a linear map connecting pKd to pChEMBL
    for _m in ("covalent", "noncovalent"):                                                          # Loop over both docking modes to print their respective calibration statistics
        _f = (_cal or {}).get(_m) or {}                                                             # Safely extract the fitted dictionary for the current mode
        if _f.get("a") is not None:                                                                 # Check if valid slope/intercept parameters were successfully fitted
            print(f"[stage3]   {_m}: R2={_f['r2']:.3f} spearman={_f['spearman']:.3f} n={_f['n']} "
                  f"<- how well docking tracks measured affinity (bounds what AL can learn)")       # Report tracking quality to bound expectations on what the AL loop can actually learn

    # -----------------------------------------------------------------------------------------
    # Generator Setup & Target-Conditioning: Instantiate the GNN agent faithfully and inject 
    # the biological pocket context via FiLM.
    # -----------------------------------------------------------------------------------------
    # Build the generator policy ONCE, faithfully, before the rounds begin.
    #   - hidden_dim/edge_dim MUST match the values the checkpoint was trained with (Stage-2 used
    #     hidden_dim=128, edge_dim=4). The old default of 64 silently shape-mismatched the load.
    #   - pocket_vec is registered non-persistent, so it is NOT inside the checkpoint. Target-aware
    #     generation therefore requires re-installing the ESM pocket embedding via set_pocket(),
    #     exactly as train_rl.py does after loading — otherwise FiLM conditioning is all-zeros and
    #     the policy generates as if it had never seen the KRAS pocket.
    
    # Dimensionality of the pocket embedding is implied by the ESM model in the config (ESM2 t33 = 1280, otherwise 0).
    _pdim = cfg_pocket_dim(cfg)                                                                     # Resolve the pocket embedding dimensionality implied by the ESM model in the config
    # Instantiate the GNN actor/critic agent with the exact trained structural widths and optional pocket conditioning
    agent = MoleculeAgent(node_dim, num_actions, hidden_dim=hidden_dim, edge_dim=edge_dim, pocket_dim=_pdim) 
    # Load the Stage-2 policy checkpoint (i.e. trained Stage-2 generator weights) into host RAM regardless of final target device to 
    _ckpt = torch.load(agent_ckpt, map_location="cpu")                                              
    # Extract the nested model state dictionary regardless of checkpoint formatting
    _sd = _ckpt["model"] if "model" in _ckpt else _ckpt                                             
    
    # Shape-safe state dictionary loading logic:
    # Stage-3 deepened the critic network. A strict load would crash due to shape mismatches.
    # We dynamically filter the state dictionary to load only matching actor/encoder tensors, skipping the 
    # incompatible value head entirely (which is completely unused during inference/sampling anyway).
    _mine = agent.state_dict()                                                                      # Retrieve the uninitialized target state dictionary of the current architecture
    _compat = {k: v for k, v in _sd.items() if k in _mine and _mine[k].shape == v.shape}            # Filter for weights that exist in both models and have identical dimensional shapes
    _skipped = [k for k in _sd if k not in _compat]                                                 # Track which specific tensors were omitted to provide transparency
    agent.load_state_dict(_compat, strict=False)                                                    # Load the filtered compatible weights safely without throwing strictness errors
    print(f"[stage3] loaded {len(_compat)}/{len(_sd)} checkpoint tensors; skipped (shape mismatch, e.g. deeper Stage-3 critic): {_skipped}") # Report loading details and explicitly state omitted heads
    
    _pnpy = pocket_npy or cfg.get("esm_pocket_embedding")                                           # Prefer an explicit pocket path, else fall back to the config's embedding location
    if _pdim > 0 and _pnpy and os.path.exists(_pnpy):                                               # Only condition when pocket embedding is enabled and its file is actually present
        agent.set_pocket(np.load(_pnpy))                                                            # Re-install the fixed target pocket vector so FiLM matches the trained behaviour
        print(f"[stage3] pocket conditioning restored (dim={_pdim}) from {_pnpy}")                  # Echo the restored conditioning so the operator can confirm target-awareness

    # -----------------------------------------------------------------------------------------
    # Active Learning Loop (Generate -> Acquire -> Oracle -> Label -> Retrain)
    # -----------------------------------------------------------------------------------------
    for r in range(rounds):                                                                         # Iterate sequentially through the requested active learning macro-epochs
        print(f"\n===== ACTIVE-LEARNING ROUND {r} =====")                                           # Output visual separator and current round tracker to stdout
        
        # RESUME GUARD: if this round already completed in a prior (possibly killed or cross-account)
        # session, skip all of its expensive work and simply advance the pointers. A round counts as
        # done when its metrics file and its retrained surrogate both exist on disk.
        done_metrics = os.path.join(out_dir, f"metrics_round{r}.json")                              # Path to this round's summary metrics (written last -> proof of completion)
        done_surr = os.path.join(out_dir, f"surrogate_round{r+1}")                                  # Path to this round's retrained surrogate ensemble directory
        done_csv = os.path.join(out_dir, f"surrogate_train_round{r+1}.csv")                         # Path to this round's augmented training CSV
        if os.path.exists(done_metrics) and os.path.exists(os.path.join(done_surr, "metrics.json")): # Both present -> round is genuinely complete
            print(f"[round {r}] already complete -> skipping (resume)")                             # Announce the skip so the operator sees the resume taking effect
            train_csv, curr_surrogate = done_csv, done_surr                                         # Advance the state pointers exactly as the normal path would
            continue                                                                                # Jump straight to the next, not-yet-computed round
            
        # 1) GENERATE (policy already built + conditioned above; just sample this round's pool)
        smis = list({_canon(s) for s in sample_smiles(agent, make_env, pool, device=device) if _canon(s)}) # Sample molecules, canonicalize, and wrap in a set to enforce uniqueness
        print(f"[round {r}] generated {len(smis)} unique molecules")                                # Log total unique structural yields from the GNN policy

        # 2) ACQUIRE (surrogate score -> UCB -> diverse batch)
        sc = AffinityScorer(curr_surrogate, device=device)                                          # Stage-3 accel: score the generated pool on the GPU (identical outputs, far faster than CPU)
        mols = [Chem.MolFromSmiles(s) for s in smis]                                                # Bulk parse the generated SMILES strings into RDKit representations
        mean, unc = sc.score_mols(mols)                                                             # Compute epistemic uncertainty and mean affinity predictions across the deep ensemble
        idx = select_batch(smis, np.array(mean), np.array(unc), n_dock=n_dock, round_idx=r, n_rounds=rounds) # Select a diverse, high-confidence acquisition batch using Upper Confidence Bound (UCB)
        batch = [smis[i] for i in idx]                                                              # Slice the original SMILES array utilizing the returned acquisition indices
        print(f"[round {r}] acquired {len(batch)} molecules to dock")                               # Log the exact number of candidates routed to the physics oracle

        # 3) ORACLE
        pose_dir = os.path.join(out_dir, f"poses_round{r}")                                         # Define a round-specific output folder for saving 3D SDF conformations
        labels = dock_batch(batch, cfg, receptor, center, size, pose_dir, cache=cache)              # Pass acquisition batch to the physics engine yielding pKd-scaled affinity labels
        print(f"[round {r}] docked {len(labels)} successfully "                                     # Log successful physics calculations
              f"({sum(1 for _, _, m in labels if m == 'covalent')} covalent to Cys12)")             # Break down logs to specifically count how many successfully attached to Cys12

        # 4) LABEL -> augment training set
        new_csv = os.path.join(out_dir, f"surrogate_train_round{r+1}.csv")                          # Define the filename for the augmented dataset meant for the next generation
        base = pd.read_csv(train_csv)                                                               # Load the accumulated historical physics knowledge base into a DataFrame
        
        # -----------------------------------------------------------------------------------------
        # Semantic Group: Calibration & Provenance Labeling
        # Map every docking score through the FROZEN, per-mode calibration. Because (a, b) never
        # change, the same molecule always receives the same label, rounds stay comparable, and the
        # absolute affinity signal survives -- none of which held under per-round z-normalisation.
        # -----------------------------------------------------------------------------------------
        _cal_rows = [(s, v, m, apply_calibration(_cal, v, m)) for s, v, m in labels]                    # Apply the pre-fitted mode-specific mathematical mapping to scale pKd up to the target labels
        add = pd.DataFrame([(s, y) for s, _, _, y in _cal_rows], columns=["smiles", base.columns[1]])   # Form a new two-column dataframe ready for pure ingestion by the surrogate training engine
        
        # Provenance is written to a SEPARATE file: the training CSV must stay exactly two columns so
        # surrogate.train_surrogate's reader is unaffected.
        pd.DataFrame(_cal_rows, columns=["smiles", "dock_score_pkd", "mode", "calibrated_label"]).to_csv(   # Construct a detailed diagnostic dataframe carrying all conversion tracing metadata
            os.path.join(out_dir, f"labels_round{r}.csv"), index=False)                                     # Save provenance metadata independently so researchers can audit the scale conversions
        pd.concat([base, add], ignore_index=True).to_csv(new_csv, index=False)                              # Concatenate historical data + newly calibrated data and flush to disk seamlessly

        # 5) RETRAIN surrogate (subprocess -> proven training code)
        new_surr = os.path.join(out_dir, f"surrogate_round{r+1}")                                   # Define target output directory for the retrained deep ensemble
        subprocess.run([sys.executable, "-m", "surrogate.train_surrogate", "--csv", new_csv,        # Invoke external python subprocess to completely isolate memory and reuse training script
                        "--out", new_surr, "--members", "5", "--epochs", "150", "--device", device],# Pass ensemble size, duration, and hardware target as CLI arguments
                       check=True)                                                                  # Enforce strict failure catching (aborts loop if training crashes)

        # 6) MEASURE: uncertainty on a fresh generated sample + surrogate metrics + docking distribution
        sc2 = AffinityScorer(new_surr, device=device)                                               # Stage-3 accel: Load the freshly trained ensemble for GPU-accelerated uncertainty measurement
        fresh = [Chem.MolFromSmiles(s) for s in smis[:500]]                                         # Sample a fresh slice of un-docked molecules from the generated pool
        _, unc2 = sc2.score_mols(fresh)                                                             # Measure the new ensemble's epistemic uncertainty over the generated domain
        metrics = json.load(open(os.path.join(new_surr, "metrics.json")))                           # Parse the training performance report generated by the subprocess
        
        # -----------------------------------------------------------------------------------------
        # Evaluation against the frozen holdout: Spearman correlation
        # Track generalizability against the static scaffold split from round 0.
        # -----------------------------------------------------------------------------------------
        _holdout_sp = None                                                                          # Initialize the fixed holdout tracking metric variable safely
        try:                                                                                        # Wrap holdout evaluation in try block to prevent minor eval errors from crashing AL
            _hdf = pd.read_csv(_holdout_csv)                                                        # Read the frozen round-0 evaluation dataset into memory
            _hmols = [Chem.MolFromSmiles(x) for x in _hdf[_hdf.columns[0]].astype(str)]             # Translate frozen SMILES back into valid RDKit topological structures
            _hmean, _ = sc2.score_mols(_hmols)                                                      # Ask the newly trained proxy ensemble to predict affinity for the holdout set
            _cmp = pd.DataFrame({"p": np.asarray(_hmean, float),                                    # Package predictions and true validation labels side-by-side into a dataframe
                                 "t": _hdf[_hdf.columns[1]].to_numpy(float)}).dropna()              # Drop any internal calculation NaNs to ensure statistical validity
            _holdout_sp = float(_cmp["p"].corr(_cmp["t"], method="spearman")) if len(_cmp) > 2 else None # Compute Spearman rank correlation over the fixed evaluation slice
        except Exception as _e:                                                                     # Intercept calculation failures (e.g. all NaNs in predictions)
            print(f"[round {r}] fixed-holdout eval skipped: {_e}")                                  # Log failure gracefully without halting the overarching AL cycle
            
        dock_scores = [v for _, v, _ in labels]                                                     # pKd-scaled docking scores (mode-tagged triples now) isolated for median tracking
        _n_cov = sum(1 for _, _, m in labels if m == "covalent")                                    # How many of this round's labels came from covalent Cys12 docking
        
        row = {"round": r,                                                                          # Construct summary dict: log the current loop iteration
               "scaffold_spearman": metrics.get("ensemble_spearman"),                               # Log intra-round test metric: rank correlation across strict chemical splits on current data
               "scaffold_rmse_z": metrics.get("ensemble_rmse_z"),                                   # Log error metric: normalized prediction variance on holdout sets
               "gen_uncertainty_mean": float(np.mean(unc2)) if len(unc2) else None,                 # Log confidence metric: average uncertainty the surrogate feels against the generator
               "dock_median": float(np.median(dock_scores)) if dock_scores else None,               # Log physics metric: the median true docking affinity captured this round
               "n_labels_added": len(labels),                                                       # Log acquisition metric: total valid labels injected into the database
               "n_covalent": _n_cov,                                                                # Log specific count of successful warhead bindings to target residues
               "n_noncovalent": len(labels) - _n_cov,                                               # Log specific count of generic affinity bindings
               "holdout_spearman": _holdout_sp}                                                     # Log longitudinal metric: Spearman correlation on the strictly frozen base test set
        
        json.dump(row, open(os.path.join(out_dir, f"metrics_round{r}.json"), "w"), indent=2)        # Flush the assembled summary dictionary to disk (this acts as the completion flag)
        print(f"[round {r}] {row}")                                                                 # Echo the summary metrics to stdout for real-time monitoring
        train_csv, curr_surrogate = new_csv, new_surr                                               # Advance the state pointers for the CSV and ensemble directory to fuel the next round

    print("\n[active-learning] done. Plot with eval/al_curves.py")                                  # Notify operator of successful termination and point to evaluation scripts


def cfg_pocket_dim(cfg):
    """
    Determines the expected dimensionality of the ESM protein language model embedding.
    
    Args:
        cfg (dict): The master configuration dictionary.
        
    Returns:
        int: 1280 if the ESM2 t33 model is specified, otherwise 0.
    """
    # -----------------------------------------------------------------------------------------
    # Configuration Resolution
    # -----------------------------------------------------------------------------------------
    return 1280 if cfg.get("esm_model", "").startswith("esm2_t33") else 0                           # Resolve 1280 for standard ESM2 650M parameters, defaulting to 0 for unconditioned runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent_ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_csv", required=True)
    ap.add_argument("--out_dir", default="../artifacts/active_learning")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--surrogate_dir", default="../artifacts/surrogate_kras")
    ap.add_argument("--physdock", default=None)
    args = ap.parse_args()
    raise SystemExit("Use the ready-made driver instead: `cd src && python run_stage3.py --help`. " # Halt direct execution and redirect to the properly wired stage-3 driver script
                     "It wires make_env(), node_dim, num_actions faithfully and calls run(...).")   # Explain that the driver handles complex environment initialization safely