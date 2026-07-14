"""
======================================
Active-Learning Orchestrator (Stage 3)
======================================
 
This script acts as the closed-loop active-learning orchestrator for Stage 3 of the pipeline.
It bridges the deep generative policy (the GNN agent) with the computationally expensive 
physics-based oracle (Gnina docking) using a fast surrogate model as a proxy.

It runs R rounds of the proxy <-> oracle cycle:
  1. GENERATE   sample a large pool of molecules from the current policy (PPO agent)
  2. ACQUIRE    surrogate-score the pool, pick an information-rich, chemically diverse batch using UCB
  3. ORACLE     covalently dock the batch to Cys12 (gnina-GPU) + PoseBusters + interaction filter (ProLIF to check if the ligand actually engages with the switch-II pocket selectivity residues)
  4. LABEL      append (SMILES, oracle score) to the surrogate training CSV (z-normalized)
  5. RETRAIN    retrain the deep ensemble on the augmented set (subprocess -> train_surrogate.py)
  6. MEASURE    log scaffold-split Spearman, calibration, uncertainty-on-generated, and top-k oracle score

The four measured curves (written to metrics_round_*.json) act as the empirical proof that 
the loop works: they should improve monotonically across rounds. 

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
from surrogate.predict import AffinityScorer                      # noqa: E402
from activelearn.acquire import select_batch                      # noqa: E402
from oracle.dock import dock_many, dock_many_fast, which_engine   # noqa: E402
from oracle.covalent_dock import covalent_dock_many               # noqa: E402
from oracle.prepare_receptor import ligand_com                    # noqa: E402
from oracle.cache import DockCache                                # noqa: E402  Stage-3 accel: disk cache -> free repeats + cross-account resume
from reward.warhead import has_warhead                            # noqa: E402


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
    Routes a batch of molecules to the appropriate docking protocol (covalent vs non-covalent).
    
    Splits the incoming batch of SMILES into two sub-pools based on the presence of a 
    pre-defined covalent warhead. Molecules with warheads are routed to `covalent_dock_many` 
    (targeting Cys12). The rest are routed to `dock_many_fast` for standard non-covalent 
    affinity prediction. The results are unified and affinities are negated to align with 
    pChEMBL maximization targets.
    
    A shared DockCache (when provided) makes any molecule already docked under these parameters —
    across rounds or across a resumed session - return instantly, at zero cost to accuracy since 
    every dock is seed-locked.
    
    Args:
        smiles (list): List of canonical SMILES strings to evaluate.
        cfg (dict): Configuration dictionary containing docking parameters.
        receptor (str): Filepath to the receptor structure.
        center (tuple): 3D coordinates (x, y, z) for the center of the docking box.
        size (tuple): Dimensions of the docking bounding box in Angstroms.
        pose_dir (str): Directory to save the output SDF pose files.
        cache (DockCache, optional): Disk-backed JSONL cache to accelerate repeated docking.
        
    Returns:
        list: A list of tuples containing (smiles, score) where score = -affinity (higher = better).
    """
    # ----------------------------------------------------------------------------------------------
    # Warhead Detection & Routing: Separate the batch based on reactive capability (covalent vs non)
    # for proper physics simulation, and extract the covalent target specification from the config.
    # ----------------------------------------------------------------------------------------------
    cov = [s for s in smiles if has_warhead(Chem.MolFromSmiles(s))]                                 # Filter SMILES containing a defined reactive covalent warhead motif
    noncov = [s for s in smiles if s not in set(cov)]                                               # Collect all remaining SMILES to be processed as standard non-covalent ligands
    rows = []                                                                                       # Initialize an empty accumulator for the unified scoring results
    cys = cfg.get("covalent", {}).get("cys_spec", "A:12:SG")                                        # Extract covalent target specification (defaulting to KRAS Cys12 sulfur)
    
    # ------------------------------------------------------------------------------------------------------
    # Covalent Docking Execution: If the covalent pool is non-empty, route to the covalent docking routine,
    # and if the docking completes successfully, append the (SMILES, inverted affinity) to the results.
    # ------------------------------------------------------------------------------------------------------
    if cov and which_engine("gnina") == "gnina":                                                    # Verify covalent candidates exist and that gnina is successfully installed in the PATH
        for r in covalent_dock_many(cov, receptor, cys, engine="gnina", keep_pose_dir=pose_dir,     # Dispatch warhead molecules to the specialized covalent grid routine
                                    cache=cache):                                                   # Pass the disk-cache to intercept previously computed identical poses
            if r.ok and r.affinity is not None:                                                     # Validate that the docking completed without geometric or systemic failure
                rows.append((r.smiles, -r.affinity))                                                # Invert affinity (lower energy = stronger binding -> higher score) and append
                
    # -------------------------------------------------------------------------------------------------------------
    # Non-Covalent Docking Execution: If the non-covalent pool is non-empty, route to the standard docking routine,
    # and if the docking completes successfully, append the (SMILES, inverted affinity) to the results.
    # -------------------------------------------------------------------------------------------------------------
    for r in dock_many_fast(noncov, receptor, center, size,                                         # Dispatch the non-reactive pool to standard localized box docking
                            engine=cfg["docking"]["engine"], gpu=cfg["docking"].get("gpu", True),   # Read target engine and GPU acceleration flags from the master config
                            progress=True, cache=cache):                                            # Enable progress bar logging and attach the identical disk-cache
        if r.ok and r.affinity is not None:                                                         # Validate successful structural calculation
            rows.append((r.smiles, -r.affinity))                                                    # Invert the scalar affinity value to maximize the reward gradient and append
    return rows                                                                                     # Return the fully compiled and aligned list of tuple scores


def run(agent_ckpt, make_env, node_dim, num_actions, config, base_csv, out_dir,
        rounds=3, pool=3000, n_dock=250, device="cuda", surrogate_dir="../artifacts/surrogate_kras",
        physdock_repo=None, hidden_dim=128, edge_dim=4, pocket_npy=None):
    """
    Executes the multi-round active learning loop connecting generation, scoring, and retraining.
    
    Loads the trained PyG actor-critic agent and restores its ESM pocket embedding condition. 
    It then loops through predefined `rounds`. In each round, it generates thousands of candidate 
    SMILES, uses the surrogate ensemble to select a diverse high-confidence subset (via UCB), 
    evaluates that subset using actual Gnina docking, and appends the physics-backed labels 
    to the training set. It then shells out to train a fresh deep ensemble surrogate, records 
    performance metrics, and repeats.
    
    Robustness features include a resume guard (skips completed rounds if execution is interrupted)
    and a disk-backed JSONL docking cache (eliminates redundant physics calculations).
    
    Args:
        agent_ckpt (str): Path to the trained Stage-2 GNN generator weights.
        make_env (callable): Function to instantiate the vectorized RL environment.
        node_dim (int): Number of node features for the agent input.
        num_actions (int): Size of the action vocabulary.
        config (str): Path to the master YAML config defining the physics bounds.
        base_csv (str): Path to the initial surrogate training dataset.
        out_dir (str): Output directory for models, poses, and metrics.
        rounds (int): Total active learning loops to execute.
        pool (int): Molecules to generate per round.
        n_dock (int): Number of molecules to acquire and dock per round.
        device (str): Compute device ("cpu" or "cuda").
        surrogate_dir (str): Path to the currently active proxy ensemble.
        physdock_repo (str): Path to optional PhysDock physics repository.
        hidden_dim (int): Hidden dimension size the agent was trained with.
        edge_dim (int): Edge dimension size the agent was trained with.
        pocket_npy (str): Path to the ESM pocket embedding vector file.
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

    # A single disk-backed docking cache, living inside out_dir, makes re-encountered molecules free 
    # (across rounds and across resumed sessions), and because it is just a JSONL file it travels with the rest of the resume state.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")                                        # Silence HF tokenizer fork warnings when ESM is imported downstream
    # Open (or resume) the persistent JSONL docking cache to bypass duplicate compute.
    cache = DockCache(os.path.join(out_dir, "dock_cache.jsonl"))                                    

    # -----------------------------------------------------------------------------------------
    # Generator Setup & Target-Conditioning: Instantiate the GNN agent faithfully and inject 
    # the biological pocket context via FiLM.
    # -----------------------------------------------------------------------------------------
    # Lazy imports of repo agent + sampler
    import torch                                                                                    # Import PyTorch locally to avoid heavy memory allocation during CLI --help checks
    from gnn_agent import MoleculeAgent                                                             # Import the architectural blueprint for the structural generator
    from activelearn.generate import sample_smiles                                                  # Import the inference routine for batch SMILES generation

    train_csv = os.path.join(out_dir, "surrogate_train_round0.csv")                                 # Define the filename for the initial round 0 starting dataset
    pd.read_csv(base_csv).to_csv(train_csv, index=False)                                            # Copy the base training data over to the active learning working directory
    curr_surrogate = surrogate_dir                                                                  # Initialize pointer to track the most recently trained ensemble directory

    # Build the generator policy once, faithfully, before the rounds begin.
    # - hidden_dim/edge_dim must match the values the checkpoint was trained with (Stage-2 used hidden_dim=128, edge_dim=4). The old default of 64 silently shape-mismatched the load.
    #   
    # - pocket_vec is registered non-persistent, so it is not inside the checkpoint. Target-aware generation therefore requires re-installing the ESM pocket embedding via set_pocket(),
    #   exactly as train_rl.py does after loading — otherwise FiLM conditioning is all-zeros and the policy generates as if it had never seen the KRAS pocket.
    
    # Dimensionality of the pocket embedding is implied by the ESM model in the config (ESM2 t33 = 1280, otherwise 0).
    _pdim = cfg_pocket_dim(cfg)                                                                     # Resolve the pocket embedding dimensionality implied by the ESM model in the config
    # Instantiate the GNN actor/critic agent with the exact trained structural widths and optional pocket conditioning
    agent = MoleculeAgent(node_dim, num_actions, hidden_dim=hidden_dim, edge_dim=edge_dim, pocket_dim=_pdim) 
    # Load the Stage-2 policy checkpoint (i.e. trained Stage-2 generator weights) into host RAM regardless of final target device to 
    _ckpt = torch.load(agent_ckpt, map_location="cpu")                                              
    agent.load_state_dict(_ckpt["model"] if "model" in _ckpt else _ckpt, strict=False)              # Load tensors into agent; strict=False tolerates extra Stage-3 critic heads if present
    # Resolve the path to the ESM pocket embedding vector file, prioritizing explicit argument over config
    _pnpy = pocket_npy or cfg.get("esm_pocket_embedding")                                           # Resolve path to pocket vector file (explicit arg prioritized over config)
    # If the pocket embedding is requested and the file exists, load it into the agent to restore target-awareness for FiLM modulation
    if _pdim > 0 and _pnpy and os.path.exists(_pnpy):                                               # Verify pocket conditioning is requested and the physical embedding file is available
        agent.set_pocket(np.load(_pnpy))                                                            # Re-install the fixed target pocket vector so FiLM modulates identically to training
        print(f"[stage3] pocket conditioning restored (dim={_pdim}) from {_pnpy}")                  # Echo the successful restoration of target-awareness to the console

    # -----------------------------------------------------------------------------------------
    # Active Learning Loop (Generate -> Acquire -> Oracle -> Label -> Retrain)
    # -----------------------------------------------------------------------------------------
    for r in range(rounds):                                                                         # Iterate sequentially through the requested active learning macro-epochs
        print(f"\n===== ACTIVE-LEARNING ROUND {r} =====")                                           # Output visual separator and current round tracker to stdout
        
        # RESUME GUARD: if this round already completed in a prior (possibly killed or interrupted)
        # session, skip all of its expensive work and simply advance the pointers. A round counts as
        # done when its metrics file and its retrained surrogate both exist on disk.
        done_metrics = os.path.join(out_dir, f"metrics_round{r}.json")                              # Path to this round's summary metrics (written last -> absolute proof of completion)
        done_surr = os.path.join(out_dir, f"surrogate_round{r+1}")                                  # Path to this round's downstream retrained proxy ensemble directory
        done_csv = os.path.join(out_dir, f"surrogate_train_round{r+1}.csv")                         # Path to this round's completed and augmented training dataset CSV
        if os.path.exists(done_metrics) and os.path.exists(os.path.join(done_surr, "metrics.json")): # Check if both artifacts exist verifying the round completed entirely without crashing
            print(f"[round {r}] already complete -> skipping (resume)")                             # Announce the resume skip so the operator sees the acceleration taking effect
            train_csv, curr_surrogate = done_csv, done_surr                                         # Fast-forward the dataset and ensemble pointers exactly as the normal loop would
            continue                                                                                # Jump straight to the next evaluation round

        # 1) GENERATE (policy already built + conditioned above; just sample this round's pool)
        smis = list({_canon(s) for s in sample_smiles(agent, make_env, pool, device=device) if _canon(s)}) # Sample molecules, canonicalize, and wrap in a set to enforce uniqueness
        print(f"[round {r}] generated {len(smis)} unique molecules")                                # Log total unique structural yields from the GNN policy

        # 2) ACQUIRE (surrogate score -> UCB -> diverse batch)
        sc = AffinityScorer(curr_surrogate, device=device)                                          # Stage-3 accel: Initialize the current surrogate ensemble for GPU-accelerated batch scoring
        mols = [Chem.MolFromSmiles(s) for s in smis]                                                # Bulk parse the generated SMILES strings into RDKit representations
        mean, unc = sc.score_mols(mols)                                                             # Compute epistemic uncertainty and mean affinity predictions across the deep ensemble
        idx = select_batch(smis, np.array(mean), np.array(unc), n_dock=n_dock, round_idx=r, n_rounds=rounds) # Select a diverse, high-confidence acquisition batch using Upper Confidence Bound (UCB)
        batch = [smis[i] for i in idx]                                                              # Slice the original SMILES array utilizing the returned acquisition indices
        print(f"[round {r}] acquired {len(batch)} molecules to dock")                               # Log the exact number of candidates routed to the physics oracle

        # 3) ORACLE
        pose_dir = os.path.join(out_dir, f"poses_round{r}")                                         # Define a round-specific output folder for saving 3D SDF conformations
        labels = dock_batch(batch, cfg, receptor, center, size, pose_dir, cache=cache)              # Pass acquisition batch to the physics engine yielding ground-truth affinity labels
        print(f"[round {r}] docked {len(labels)} successfully")                                     # Log successful physics calculations (accounting for geometric failures)

        # 4) LABEL -> augment training set
        new_csv = os.path.join(out_dir, f"surrogate_train_round{r+1}.csv")                          # Define the filename for the augmented dataset meant for the next generation
        base = pd.read_csv(train_csv)                                                               # Load the accumulated historical physics knowledge base into a DataFrame
        add = pd.DataFrame([(s, v) for s, v in labels], columns=["smiles", base.columns[1]])        # Convert new batch labels into an aligned dataframe structure
        
        # per-source z-norm of docking labels before merge (higher=better already)
        if len(add):                                                                                # Check if the newly acquired dataframe is populated
            v = add[add.columns[1]].to_numpy(float)                                                 # Extract the raw affinity scalar column as a numpy float array
            add[add.columns[1]] = (v - v.mean()) / (v.std() + 1e-8)                                 # Standardize the new batch to mean 0 / std 1 to stabilize surrogate gradient descent
        pd.concat([base, add], ignore_index=True).to_csv(new_csv, index=False)                      # Concatenate historical (i.e., already present in the surrogate train dataset) + new data and flush to disk without row indices

        # 5) RETRAIN surrogate (subprocess -> proven training code)
        new_surr = os.path.join(out_dir, f"surrogate_round{r+1}")                                   # Define target output directory for the retrained deep ensemble
        subprocess.run([sys.executable, "-m", "surrogate.train_surrogate", "--csv", new_csv,        # Invoke external python subprocess to completely isolate memory and reuse training script
                        "--out", new_surr, "--members", "5", "--epochs", "150", "--device", device],# Pass ensemble size, duration, and hardware target as CLI arguments
                       check=True)                                                                  # Enforce strict failure catching (aborts loop if training crashes)

        # 6) MEASURE: uncertainty on a fresh generated sample of candidates produced in 1 + surrogate metrics + docking distribution
        sc2 = AffinityScorer(new_surr, device=device)                                               # Stage-3 accel: Load the freshly trained ensemble for GPU-accelerated uncertainty measurement
        fresh = [Chem.MolFromSmiles(s) for s in smis[:500]]                                         # Sample a fresh slice of un-docked molecules from the generated pool
        _, unc2 = sc2.score_mols(fresh)                                                             # Measure the new ensemble's epistemic uncertainty over the generated domain
        metrics = json.load(open(os.path.join(new_surr, "metrics.json")))                           # Parse the training performance report generated by the subprocess
        dock_scores = [v for _, v in labels]                                                        # Isolate the raw scalar docking scores from the current batch
        
        row = {"round": r,                                                                          # Construct summary dict: log the current loop iteration
               "scaffold_spearman": metrics.get("ensemble_spearman"),                               # Log generalizability metric: rank correlation across strict chemical splits
               "scaffold_rmse_z": metrics.get("ensemble_rmse_z"),                                   # Log error metric: normalized prediction variance on holdout sets
               "gen_uncertainty_mean": float(np.mean(unc2)) if len(unc2) else None,                 # Log confidence metric: average uncertainty the surrogate feels against the generator
               "dock_median": float(np.median(dock_scores)) if dock_scores else None,               # Log physics metric: the median true docking affinity captured this round
               "n_labels_added": len(labels)}                                                       # Log acquisition metric: total valid labels injected into the database
        
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