"""
================================================
Stage-3 Runner — Closed-Loop Active Learning (RL)
================================================

This is the single, runnable entry point for the Stage-3 closed-loop active learning loop. 
The core logic resides in `activelearn/loop.py`, which deliberately ships without a `__main__` block.
This driver script bridges that gap by flawlessly reconstructing the Stage-2 training environment 
before execution. If the environment factory or the exact action-space dimensions do not match 
the Stage-2 checkpoint exactly, the policy load becomes meaningless.

What this script wires together to prevent configuration drift:
    - ActionSpec(max_atoms=25): Defines the flat discrete action space (`num_actions`).
    - node_dim: Computed as `len(atom_types) + 3 + 3`, aligning with the GNN input width the checkpoint expects.
    - hidden_dim=128, edge_dim=4: The explicitly trained network widths (overriding any default class values like 64).
    - make_env(): A factory that produces a clean `MoleculeEnvironment` for rapid sampling. It deliberately 
      turns off affinity and diversity scoring to skip expensive per-step oracle calls during pure generation.
    - pocket_embedding: Extracts the ESM pocket vector from the config and installs it so that 
      the agent's FiLM layers are correctly target-conditioned (matching training).

Usage:
Run this script from the `src/` directory to maintain repo conventions:
    $ cd src
    $ python run_stage3.py \
        --config ../configs/kras_g12c.yaml \
        --agent_ckpt ../artifacts/stage2_policy/policy.pt \
        --base_csv ../data/surrogate_train.csv \
        --surrogate_dir ../artifacts/surrogate_kras \
        --out_dir ../artifacts/active_learning \
        --rounds 3 --pool 3000 --n_dock 250

Note: The `--out_dir` is designed to be a STABLE, resumable directory (no per-launch timestamp). 
Re-executing the exact same command after an interruption or on a fresh AWS account will skip 
any round already completed and reuse the on-disk docking cache automatically.
"""

import argparse
import os
import sys

# -----------------------------------------------------------------------------------------
# System Path Configuration
# Ensure that local project modules can be resolved regardless of the launch directory.
# -----------------------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                      # Prepend the src/ directory to the path so internal modules (`chem_env`, `gnn_agent`) resolve identically to train_rl.py

import yaml                                                                                         # Import the YAML parser to ingest the universal project configuration file

from chem_env import ActionSpec, MoleculeEnvironment                                                # Import the vocabulary specification and the single-agent graph environment
from activelearn.loop import run as run_active_learning                                             # Import the primary closed-loop engine (handles generate -> dock -> retrain)


def build_make_env(spec, device):
    """
    Constructs a zero-argument factory function that yields fresh MoleculeEnvironments.

    The active-learning sampling engine (`activelearn/generate.sample_smiles`) requires a 
    factory to create fresh environments on demand. Because the policy sampling phase ignores 
    reward signals, this factory intentionally configures the environment with affinity, 
    diversity, and warhead scoring disabled. This guarantees faithful rollout behavior 
    while bypassing the massive computational overhead of querying the surrogate thousands of times.

    Args:
        spec (ActionSpec): The shared vocabulary definition (fixes the discrete action space).
        device (torch.device): The hardware compute device designated for tensor allocations.

    Returns:
        Callable[[], MoleculeEnvironment]: A parameter-free factory returning isolated environments.

    Example:
        >>> import torch
        >>> from chem_env import ActionSpec
        >>> env_factory = build_make_env(ActionSpec(max_atoms=25), torch.device("cpu"))
        >>> env = env_factory()
    """
    # -------------------------------------------------------------------------------------
    # Environment Factory Setup
    # Safely construct the reward config by inheriting (all the required downstream) project 
    # defaults, then disable expensive oracles to ensure lightweight, rapid sampling.
    # -------------------------------------------------------------------------------------
    from reward.composite import default_reward_cfg as _drc                                         # Import the authoritative default reward configuration factory to inherit expected keys
    reward_cfg = dict(_drc())                                                                       # Instantiate a base configuration dictionary containing all requisite reward thresholds
    reward_cfg.update(use_affinity=False, use_diversity=False, use_warhead=False)                   # Explicitly disable computationally heavy scoring oracles for the rapid sampling phase
    reward_cfg.setdefault("property_ceiling", 12.0)                                                 # Ensure the property ceiling key is safely populated to prevent downstream KeyErrors
    def _make():                                                                                    # Define the internal closure that serves as the zero-argument factory
        return MoleculeEnvironment(device, max_steps=40, action_spec=spec, min_atoms=5,             # Instantiate the environment mirroring Stage-2 trajectory bounds
                                   affinity_scorer=None, diversity_archive=None, reward_cfg=reward_cfg) # Omit complex oracles to ensure ultra-fast, unhindered generation steps
    return _make                                                                                    # Return the assembled factory closure to the calling scope


def main():
    """
    Parses command-line arguments, flawlessly reconstructs the Stage-2 context, and triggers the loop.

    How it works:
    1. Ingests all necessary file paths and run-scope parameters via argparse.
    2. Resolves the compute device and rigorously re-establishes the exact `ActionSpec` 
       and dimensionalities (node, hidden, edge) used to train the original agent.
    3. Builds the lightweight environment factory using `build_make_env`.
    4. Parses the YAML config to locate the pre-computed ESM pocket embedding for FiLM conditioning.
    5. Dispatches all validated parameters to `loop.run()` to commence the active learning cycle.
    """
    # -----------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Define and capture all required paths and hyperparameters, defaulting to the repo layout.
    # -----------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Stage-3 closed-loop active learning (PPO policy).")   # Initialize the argument parser with a human-readable module description
    ap.add_argument("--config", default="../configs/kras_g12c.yaml")                                # Register the path argument for the master YAML configuration (receptor, pocket, etc.)
    ap.add_argument("--agent_ckpt", default="../artifacts/stage2_policy/policy.pt")                 # Register the path argument for the pre-trained Stage-2 PPO policy weights
    ap.add_argument("--base_csv", default="../data/surrogate_train.csv")                            # Register the path argument for the initial surrogate training dataset (round 0)
    ap.add_argument("--surrogate_dir", default="../artifacts/surrogate_kras")                       # Register the path argument for the deep-ensemble affinity predictor directory
    ap.add_argument("--out_dir", default="../artifacts/active_learning")                            # Register the path argument for the stable, timestamp-free resumable output folder
    ap.add_argument("--rounds", type=int, default=3)                                                # Register the integer argument determining total proxy-oracle acquisition cycles
    ap.add_argument("--pool", type=int, default=3000)                                               # Register the integer argument for the total molecules sampled from the policy per round
    ap.add_argument("--n_dock", type=int, default=250)                                              # Register the integer argument capping the physical docking oracle budget per round
    ap.add_argument("--device", default=None)                                                       # Register the optional argument to forcefully specify compute hardware (e.g., 'cpu' or 'cuda')
    ap.add_argument("--physdock", default=None)                                                     # Register the optional argument pointing to the external PhysDock physics refinement repo
    ap.add_argument("--max_atoms", type=int, default=25)                                            # Register the integer argument for graph size, strictly enforcing Stage-2 parity
    ap.add_argument("--hidden_dim", type=int, default=128)                                          # Register the integer argument for network width, explicitly matching the Stage-2 topology
    ap.add_argument("--edge_dim", type=int, default=4)                                              # Register the integer argument for edge feature width (e.g., bond order one-hot encoding)
    args = ap.parse_args()                                                                          # Parse the provided command-line inputs and bind them to the args namespace

    # -----------------------------------------------------------------------------------------------
    # Stage-2 Context Reconstruction
    # Dynamically calculate and lock the neural network input/output shapes to prevent load failures.
    # -----------------------------------------------------------------------------------------------
    import torch                                                                                    # Lazily import PyTorch inside the function to keep the file universally accessible
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")) # Resolve the active compute device, defaulting to CUDA if available and unspecified
    spec = ActionSpec(max_atoms=args.max_atoms)                                                     # Instantiate the action vocabulary constrained by the specified maximum atom limit
    num_actions = spec.num_actions                                                                  # Extract the total flat scalar dimension of the discrete action space
    node_dim = len(spec.atom_types) + 3 + 3                                                         # Calculate total node feature width: atom one-hot + 3 hybridization + 3 structural flags
    # Generate the dedicated zero-argument environment factory bound to this specific device
    # Instead of instantiating an object immediately, a factory returns a callable function 
    # (a closure) that can create identical objects on demand at a later time.
    make_env = build_make_env(spec, device)                                                         

    # --------------------------------------------------------------------------------------------------
    # Configuration Parsing & Conditioning
    # Open the YAML config to extract the pre-computed ESM pocket embedding for FiLM layer conditioning.
    # --------------------------------------------------------------------------------------------------
    cfg = yaml.safe_load(open(args.config))                                                         # Safely open and parse the target YAML configuration file into a python dictionary
    pocket_npy = cfg.get("esm_pocket_embedding")                                                    # Extract the filepath to the precomputed ESM pocket vector used for FiLM layer conditioning

    print(f"[run_stage3] device={device} num_actions={num_actions} node_dim={node_dim} "            # Begin printing the critical reconstructed dimensionalities to the console
          f"hidden_dim={args.hidden_dim} edge_dim={args.edge_dim}")                                 # Complete the dimension printout, allowing operators to spot pipeline mismatches instantly
    print(f"[run_stage3] out_dir={args.out_dir} (stable/resumable) rounds={args.rounds} "           # Begin printing the overarching scope of the active learning campaign
          f"pool={args.pool} n_dock={args.n_dock}")                                                 # Complete the scope printout with pool sizes and oracle budgets

    # -----------------------------------------------------------------------------------------
    # Active Learning Loop Dispatch
    # Route all assembled parameters into the core orchestration engine.
    # -----------------------------------------------------------------------------------------
    run_active_learning(                                                                            # Invoke the primary active learning closed-loop orchestrator
        agent_ckpt=args.agent_ckpt, make_env=make_env, node_dim=node_dim, num_actions=num_actions,  # Supply the core neural wiring: policy weights, environment factory, and exact shapes
        config=args.config, base_csv=args.base_csv, out_dir=args.out_dir,                           # Supply the data infrastructure: YAML config, seed labels, and the stable output directory
        rounds=args.rounds, pool=args.pool, n_dock=args.n_dock, device=str(device),                 # Supply the loop scope: acquisition iterations, sampling quotas, and hardware bindings
        surrogate_dir=args.surrogate_dir, physdock_repo=args.physdock,                              # Supply the external dependencies: the pre-trained proxy directory and physics refinement repo
        hidden_dim=args.hidden_dim, edge_dim=args.edge_dim, pocket_npy=pocket_npy)                  # Supply the final architectural constraints and target-conditioning pocket embeddings


if __name__ == "__main__":                                                                          # Guard the execution logic to ensure it only runs when explicitly invoked as a script
    main()                                                                                          # Dispatch the main function, kicking off the Stage-3 pipeline