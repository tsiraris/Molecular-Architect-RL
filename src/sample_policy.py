"""
============================================
Policy Sampling and Fair Comparison Utility
============================================

This script provides an apples-to-apples evaluation pipeline to sample SMILES and rewards 
from EITHER the trained PPO policy OR the GFlowNet policy. It ensures both models are 
evaluated under the exact same `MoleculeEnvironment` and composite reward function 
(including an optional surrogate affinity term).

By decoupling the evaluation environment from the specific training loops, it ensures 
that differences in generated SMILES distributions, validity rates, and rewards are 
solely attributable to the underlying algorithms (PPO vs. GFlowNet) rather than 
environmental discrepancies. 

It handles the architectural differences natively (PPO returns logits and values; 
GFlowNet returns only logits) and writes out a deduped CSV of the generated molecules 
alongside a JSON metadata sidecar tracking crucial metrics like validity and uniqueness.

Usage (from src/):
    python sample_policy.py --kind gfn --ckpt ../artifacts/gfn_run/gfn_agent.pt \
        --surrogate ../artifacts/active_learning_big/surrogate_round6 --n 2000 --out ../results/gfn_samples.csv
    python sample_policy.py --kind ppo --ckpt ../artifacts/stage2_policy/policy.pt \
        --surrogate ../artifacts/active_learning_big/surrogate_round6 --n 2000 --out ../results/ppo_samples.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                       # Make src/ importable when run directly

import yaml                                                                                          # Parse the target config
import numpy as np                                                                                   # Load the pocket embedding
import torch                                                                                         # Policy inference
import pandas as pd                                                                                  # Write the samples CSV

from chem_env import ActionSpec, MoleculeEnvironment                                                 # Action space + generation environment
from gflownet.train_gfn import _obs_to_data                                                          # Proven obs -> (x, ei, ea) converter shared with training
from gflownet.agent import GFlowNetAgent                                                             # GFlowNet policy
from gnn_agent import MoleculeAgent                                                                  # PPO policy


def load_agent(kind, ckpt, node_dim, num_actions, pocket_dim, pocket_npy, dev):
    """
    Loads either the GFlowNet or PPO policy from a checkpoint safely, handling shape mismatches.

    Depending on the specified `kind`, it instantiates the correct architecture. For PPO, 
    it strictly filters incoming tensors to match the current architecture shapes (ignoring 
    deeper Stage-3 critic weights if loading a Stage-2 checkpoint). It then reinstalls the 
    ESM pocket embedding for target conditioning (FiLM) if required by the model.

    Args:
        kind (str): "ppo" or "gfn", dictating which network architecture to spawn.
        ckpt (str): Path to the saved model weights (.pt file).
        node_dim (int): Dimensionality of the GNN input node features.
        num_actions (int): Flat action-space size dictated by the ActionSpec.
        pocket_dim (int): Pocket embedding width (e.g., 1280 for PPO, 0 for GFlowNet).
        pocket_npy (str or None): Filepath to the serialized ESM pocket embedding numpy array.
        dev (torch.device): Target compute device for the network.

    Returns:
        nn.Module: The loaded actor network, set to evaluation mode and moved to `dev`.
        
    Example:
        >>> agent = load_agent("ppo", "weights.pt", 12, 119, 1280, "pocket.npy", torch.device("cpu"))
    """
    # ------------------------------------------------------------------------------------------------------------
    # GFlowNet Initialization: Instantiate the GFlowNet agent, load the checkpoint (trained weights),
    # filter current tensors to match shapes, load the actor weights, and restore the pocket embedding if present.
    # ------------------------------------------------------------------------------------------------------------
    if kind == "gfn":                                                                                # GFlowNet path
        ag = GFlowNetAgent(node_dim, num_actions, 128, pocket_dim)                                   # Match the checkpoint's conditioning (1280 when warm-started from PPO, else 0)
        sd = torch.load(ckpt, map_location="cpu")                                                    # Load the trained GFlowNet weights
        mine = ag.state_dict()                                                                       # Current architecture's tensors
        compat = {k: v for k, v in sd.items() if k in mine and mine[k].shape == v.shape}             # Shape-filter defensively
        ag.load_state_dict(compat, strict=False)                                                     # Install the policy
        if pocket_dim > 0 and pocket_npy and os.path.exists(pocket_npy):                             # Restore the same target conditioning used in training
            ag.set_pocket(np.load(pocket_npy))                                                       # Install the fixed pocket vector
            
    # -------------------------------------------------------------------------------------------------------------------
    # PPO Initialization: Instantiate the PPO agent, load the checkpoint (trained weights), filter current tensors to 
    # match shapes (ignoring deeper critic weights), load the actor weights, and restore the pocket embedding if present.
    # -------------------------------------------------------------------------------------------------------------------
    else:                                                                                            # PPO path
        ag = MoleculeAgent(node_dim, num_actions, hidden_dim=128, edge_dim=4, pocket_dim=pocket_dim) # Exact trained widths
        sd = torch.load(ckpt, map_location="cpu")                                                    # Load the checkpoint
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd                                     # Unwrap {"model": ...} if present
        mine = ag.state_dict()                                                                       # Current architecture's tensors
        compat = {k: v for k, v in sd.items() if k in mine and mine[k].shape == v.shape}             # Keep only shape-matching tensors (skip deeper critic)
        ag.load_state_dict(compat, strict=False)                                                     # Load the actor/encoder faithfully
        if pocket_dim > 0 and pocket_npy and os.path.exists(pocket_npy):                             # Restore target conditioning
            ag.set_pocket(np.load(pocket_npy))                                                       # Re-install the fixed pocket vector
            
    return ag.to(dev).eval()                                                                         # Eval mode on device


def rollout(agent, env, is_gfn, dev):
    """
    Executes a single generational episode inside the environment using the provided policy.

    Resets the environment, then iteratively converts observations to PyG tensors, passes 
    them through the network, masks out illegal chemical actions, and samples from the 
    resulting probability distribution. It terminates when the model outputs the 'stop' 
    action or hits the step limit, returning the final SMILES and the calculated reward.

    Args:
        agent (nn.Module): The loaded policy network (PPO or GFlowNet).
        env (MoleculeEnvironment): The shared chemical simulation environment.
        is_gfn (bool): Flag to handle architectural return formats (logits vs. logits+value).
        dev (torch.device): Compute device for tensor operations.

    Returns:
        Tuple[str or None, float]: The canonical SMILES string (or None if chemically invalid), 
        and the terminal composite reward evaluated by the environment.
        
    Example:
        >>> smi, rew = rollout(agent, env, is_gfn=False, dev=torch.device("cpu"))
    """
    # ---------------------------------------------------------------------------------------------
    # Episode Initialization
    # Wipe the environment and prepare internal loop trackers.
    # ---------------------------------------------------------------------------------------------
    obs = env.reset()                                                                                # Fresh episode
    done, steps, reward = False, 0, 0.0                                                              # Episode state
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Inference & Generation Loop: Traverse the graph space step-by-step until termination ('stop') or a safety timeout (64 steps).
    # Pass the observation through the GNN agent, mask illegal actions, and sample proportionally from the policy.
    # ------------------------------------------------------------------------------------------------------------------------------
    while not done and steps < 64:                                                                   # Hard fail-safe on steps
        x, ei, ea = _obs_to_data(obs, dev)                                                           # Standardise the observation into graph tensors
        res = agent(x, ei, ea, torch.zeros(x.size(0), dtype=torch.long, device=dev))                 # One forward pass (dummy single-graph batch vector)
        logits = res if is_gfn else res[0]                                                           # GFlowNet -> logits; PPO -> (logits, value)
        
        if logits.dim() == 2:                                                                        # Squeeze a leading batch dim if present
            logits = logits[0]                                                                       # -> shape [A]
            
        mask = torch.as_tensor(env.get_action_mask(), device=dev)                                    # Legal-action mask
        masked = torch.where(mask.bool(), logits, torch.full_like(logits, torch.finfo(logits.dtype).min))  # Mask illegal actions
        action = int(torch.multinomial(torch.softmax(masked, -1), 1))                                # Sample proportionally to the masked policy
        
    # ----------------------------------------------------------------------------------------------
    # State Transition: Apply the selected action, step the environment, and return the 
    # final SMILES (if the episode terminated or else None) and reward.
    # ----------------------------------------------------------------------------------------------
        obs, reward, done, _ = env.step(action, curriculum_ratio=1.0)                                # Step; capture the (terminal) reward
        steps += 1                                                                                   # Advance the step counter
        
    smi = env.get_smiles() if done else None                                                         # Harvest SMILES only on proper termination
    return (None if smi in (None, "INVALID") else smi), float(reward)                                # Drop invalids


def main():
    """
    Command-line entry point for policy sampling (PPO vs. GFlowNet) and evaluation orchestration.
    
    How it works:
    1. Parses arguments to define the model type, weights, constraints, and output paths.
    2. Initializes the `MoleculeEnvironment` and optionally binds a surrogate affinity model.
    3. Loads the specified agent onto the target device.
    4. Rolls out `N` episodes, logging progress and tracking chemical validity.
    5. Deduplicates the valid results and dumps them to a CSV.
    6. Aggregates a comprehensive JSON metadata file containing rigorous validity/uniqueness stats.
    """
    # ---------------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Configure the exact parameters for the sampling run.
    # ---------------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Sample smiles+reward from a PPO or GFlowNet policy.")
    ap.add_argument("--kind", choices=["ppo", "gfn"], required=True)                                 # Which policy
    ap.add_argument("--ckpt", required=True)                                                         # Checkpoint path
    ap.add_argument("--config", default="../configs/kras_g12c.yaml")                                 # Target config
    ap.add_argument("--surrogate", default=None, help="attach surrogate for the affinity reward term")  # Consistent scoring for both policies
    ap.add_argument("--n", type=int, default=2000)                                                   # Episodes to sample
    ap.add_argument("--out", required=True)                                                          # Output CSV
    ap.add_argument("--device", default=None)                                                        # Force device
    ap.add_argument("--max_atoms", type=int, default=25)                                             # Must match the checkpoints
    a = ap.parse_args()

    # ---------------------------------------------------------------------------------------------
    # Configuration & Environment Setup
    # Resolve the hardware, config files, and build the shared chemical environment.
    # ---------------------------------------------------------------------------------------------
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))                 # Resolve device
    cfg = yaml.safe_load(open(a.config))                                                             # Load config
    spec = ActionSpec(max_atoms=a.max_atoms)                                                         # Rebuild the action space
    node_dim = len(spec.atom_types) + 3 + 3                                                          # GNN input width (== train_rl.py)

    from reward.composite import default_reward_cfg as drc                                           # Real default reward config
    rc = dict(drc())                                                                                 # Copy it
    rc.setdefault("property_ceiling", 12.0)                                                          # Ensure the key combine() needs is present
    rc["use_affinity"] = bool(a.surrogate)                                                           # Enable the affinity term only if a surrogate is attached

    scorer = None                                                                                    # Optional affinity scorer
    if a.surrogate:                                                                                  # Attach the surrogate so BOTH policies are scored identically
        from surrogate.predict import AffinityScorer                                                 # Import lazily
        scorer = AffinityScorer(a.surrogate, device=str(dev))                                        # Deep-ensemble affinity surrogate

    env = MoleculeEnvironment(dev, max_steps=40, action_spec=spec, min_atoms=5,                      # One shared env for both policies
                              affinity_scorer=scorer, reward_cfg=rc)
                              
    # ---------------------------------------------------------------------------------------------
    # Agent Loading & Rollout Execution: Load the requested policy (PPO or GFlowNet), 
    # sample N episodes, and track invalid molecules.
    # ---------------------------------------------------------------------------------------------
    pocket_dim = 1280 if a.kind == "ppo" else 0                                                      # PPO trained with pocket FiLM; GFlowNet without
    agent = load_agent(a.kind, a.ckpt, node_dim, spec.num_actions, pocket_dim,                       # Load the requested policy
                       cfg.get("esm_pocket_embedding"), dev)

    print(f"[sample_{a.kind}] device={dev} n={a.n} surrogate={'on' if scorer else 'off'}")           # Echo the run scope
    rows = []                                                                                        # Collect (smiles, reward)
    n_invalid = 0                                                                                    # Count chemically invalid terminals: validity is itself a headline metric
    
    for i in range(a.n):                                                                             # Sample N episodes
        smi, rew = rollout(agent, env, a.kind == "gfn", dev)                                         # One rollout
        if smi:                                                                                      # Keep only valid molecules
            rows.append((smi, rew))                                                                  # Record
        else:                                                                                        # Invalid terminal molecule
            n_invalid += 1                                                                           # Track it rather than silently dropping it
        if (i + 1) % 250 == 0:                                                                       # Periodic progress
            print(f"  {i+1}/{a.n} sampled, {len(rows)} valid so far")                                # Lightweight telemetry
            
    # --------------------------------------------------------------------------------------------------
    # Data Export & Metadata Tracking: Deduplicate SMILES, save the CSV, and export tracking metadata
    # such as selected kind (PPO or GFlowNet), validity, uniqueness, seed, and reward parameter weights.
    # --------------------------------------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=["smiles", "reward"]).drop_duplicates("smiles")                  # Dedupe by SMILES
    df.to_csv(a.out, index=False)                                                                    # Write the CSV
    # Validity and uniqueness are reported alongside the samples so downstream comparison never has
    # to infer them from row counts: PPO and the GFlowNet had very different validity rates (43% vs
    # 83% in earlier runs), and comparing reward/diversity on unequal-N sets would be misleading.
    meta = {"kind": a.kind, "ckpt": a.ckpt, "seed": getattr(a, "seed", None), "n_sampled": a.n,
            "n_valid": len(rows), "validity": round(len(rows) / max(1, a.n), 4),
            "n_unique": int(df.smiles.nunique()),
            "uniqueness_of_valid": round(df.smiles.nunique() / max(1, len(rows)), 4),
            "w_affinity": getattr(a, "w_affinity", None), "w_diversity": getattr(a, "w_diversity", None),
            "surrogate": a.surrogate}                                                                # Full provenance for the comparison table
            
    import json as _json                                                                             # Local import: this is the only JSON use in the module
    _json.dump(meta, open(a.out.replace(".csv", "_meta.json"), "w"), indent=2)                       # Sidecar metadata next to the samples
    print(f"[sample_{a.kind}] valid {len(rows)}/{a.n} ({meta['validity']:.1%}), "
          f"unique {meta['n_unique']} -> {a.out}")                                                   # Confirm output with the numbers that matter


if __name__ == "__main__":
    main()