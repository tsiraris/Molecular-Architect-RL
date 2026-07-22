"""
====================================
GFlowNet (Trajectory Balance) Driver
====================================

This script serves as the primary entry point for training a Generative Flow Network (GFlowNet)
using the Trajectory Balance (TB) objective. It is explicitly wired to allow a mathematically
fair comparison against the Proximal Policy Optimization (PPO) baseline.

The GFlowNet trains on the exact same Markov Decision Process (MDP), the same GNN trunk, and
critically, the same composite reward as the PPO policy. This ensures that any performance
difference is attributable to the training objective (proportional sampling vs. reward maximization)
rather than discrepancies in the problem formulation.

Two deliberate, disclosed asymmetries exist (forced by the algorithms themselves):
  1. Diversity Weight (`w_diversity` = 0): PPO's Stage-2 reward carried a Tanimoto penalty
     computed against a rolling archive of previously generated molecules. This makes R(x)
     non-stationary, violating the fixed-R(x) assumption required by GFlowNets for TB convergence.
     The comparison is thus: "PPO + explicit penalty" vs "GFlowNet proportional-sampling",
     evaluated on one common stationary objective.
  2. Warm Starting: The script initializes the GFlowNet from the trained PPO policy (`--warm_start`).
     This is standard practice in current molecular-GFlowNet work. The comparison is explicitly PPO vs. PPO-initialized GFlowNet.

Thermodynamic Beta Tuning:
BETA MUST BE CHOSEN RELATIVE TO THE REWARD SCALE. With the surrogate attached, the composite reward
spans roughly [-2, +17] (reward_scale=12). Thus, `log R = beta * (reward - baseline)` at beta=4 would
span ~76 nats against a TB residual of sigma ~4. The policy would collapse to the argmax, destroying
the mode coverage that GFlowNets exist to provide.
Rule of thumb: Pick beta so `beta * sigma(R)` is ~1-4. For sigma(R) ~ 5, beta should be {0.2, 0.5, 1.0}.
"""

# -----------------------------------------------------------------------------------------
# Standard Library & System Imports
# Import base utilities for path manipulation, argument parsing, and random seeding.
# -----------------------------------------------------------------------------------------
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                       # Make src/ importable when run directly by prepending it to the sys path

# -----------------------------------------------------------------------------------------
# External & Project Imports
# Load numerical/tensor libraries, suppress noisy logs, and import RL/GFlowNet modules.
# -----------------------------------------------------------------------------------------
import numpy as np
import torch
import yaml
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")                                                                       # Mute RDKit's benign chatter: earlier runs wrote 14M log lines of pure noise

from chem_env import ActionSpec, MoleculeEnvironment                                                 # noqa: E402  Action space + generation environment
from gflownet.train_gfn import train                                                                 # noqa: E402  Trajectory Balance trainer


def build_make_env(spec, device, cfg, scorer=None, w_affinity=0.7, w_diversity=0.0,
                   use_warhead=True, w_warhead=0.15):
    """
    Builds the environment factory using the SAME composite reward the PPO policy was trained on.

    Constructs a configuration dictionary mirroring the project's defaults, overrides specific 
    reward weights (affinity, diversity, warhead), and returns a closure (`_make`) that 
    instantiates a fresh `MoleculeEnvironment` identically configured to the PPO baseline.

    Args:
        spec (ActionSpec): Shared action-space specification defining vocabulary limits.
        device (torch.device): Compute device for allocating the environment's observation tensors.
        cfg (dict): Parsed target configuration containing runtime parameters.
        scorer (AffinityScorer or None): Deep-ensemble surrogate; when present, the affinity term is
            enabled, exactly matching PPO's Stage-2 reward. Defaults to None.
        w_affinity (float): Affinity weight. PPO's Stage-2 sweet spot was 0.7 -- match it. Defaults to 0.7.
        w_diversity (float): Diversity penalty weight. MUST be 0 for the GFlowNet: the penalty is
            computed against a rolling archive, which makes R(x) non-stationary. Defaults to 0.0.
        use_warhead (bool): Whether the covalent-warhead bonus is active. Defaults to True.
        w_warhead (float): Warhead bonus weight. Defaults to 0.15.

    Returns:
        Callable[[], MoleculeEnvironment]: A factory function producing fresh, identically-configured environments.
        
    Example:
        >>> spec = ActionSpec(max_atoms=25)
        >>> env_factory = build_make_env(spec, torch.device("cpu"), {}, w_affinity=0.7)
        >>> env = env_factory()
        >>> type(env)
        <class 'chem_env.MoleculeEnvironment'>
    """
    # -------------------------------------------------------------------------------------
    # Reward Configuration Mapping
    # Load defaults, apply strict overrides, and configure the composite reward structure.
    # -------------------------------------------------------------------------------------
    from reward.composite import default_reward_cfg as _drc                                          # The project's real defaults, so every key combine() reads is present
    rc = dict(_drc())                                                                                # Copy them into a local dictionary to prevent mutating global states
    rc.setdefault("property_ceiling", 12.0)                                                          # Guard the key chem_env normalises the property term with
    rc["use_affinity"] = scorer is not None                                                          # Affinity term on only when a surrogate is actually attached
    rc["w_affinity"] = float(w_affinity)                                                             # Match PPO's affinity weight to ensure an apples-to-apples comparison
    rc["use_diversity"] = bool(w_diversity > 0)                                                      # Off by default: see the module docstring on reward stationarity
    rc["w_diversity"] = float(w_diversity)                                                           # Explicit, so the run's objective is fully recorded
    rc["use_warhead"] = bool(use_warhead)                                                            # Covalent-warhead bonus, matched to PPO
    rc["w_warhead"] = float(w_warhead)                                                               # Its weight defining the relative importance of the warhead
    print(f"[run_gfn] reward cfg: use_affinity={rc['use_affinity']} w_affinity={rc['w_affinity']} "
          f"use_diversity={rc['use_diversity']} w_diversity={rc['w_diversity']} "
          f"use_warhead={rc['use_warhead']} reward_scale={rc['reward_scale']}")                      # Print the exact objective: this is what makes the comparison auditable

    # -------------------------------------------------------------------------------------
    # Environment Factory Closure
    # Define and return the callable that spawns identically parameterized environments.
    # -------------------------------------------------------------------------------------
    def _make():
        """Returns a fresh MoleculeEnvironment configured with the shared reward."""
        return MoleculeEnvironment(device, max_steps=40, action_spec=spec, min_atoms=5,
                                   affinity_scorer=scorer, reward_cfg=rc)                            # Same env contract PPO's vec_env builds, injecting the configured dictionary
    return _make                                                                                     # The factory closure is returned for the GFlowNet rollout orchestrator to use


def main():
    """
    CLI entry point: trains one GFlowNet at a fixed beta and seed, warm-started from PPO.
    
    Parses command-line arguments to establish the hyperparameters and paths. Forces 
    deterministic seeds across libraries. Loads the target YAML config and recreates the 
    exact PPO action space. Conditionally loads the affinity surrogate and ESM-2 pocket 
    embeddings (if warm starting from a FiLM-conditioned PPO trunk). Finally, executes 
    the Trajectory Balance training loop via `train()`.
    
    Args:
        None (Reads from sys.argv).
        
    Returns:
        None
        
    Example:
        $ python src/run_gfn.py --steps 4000 --beta 0.5 --surrogate path/to/ensemble
    """
    # -------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Define hyperparameters, directory paths, algorithm-specific weights, and system configs.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Train a GFlowNet (Trajectory Balance) on the KRAS env.")
    ap.add_argument("--config", default="../configs/kras_g12c.yaml")                                 # Target config defining target-specific parameters
    ap.add_argument("--out", default="../artifacts/gfn_run")                                         # Output directory for saving model checkpoints and logs
    ap.add_argument("--steps", type=int, default=4000)                                               # Gradient steps dictating total training duration
    ap.add_argument("--batch", type=int, default=16)                                                 # Trajectories per gradient step == parallel envs in the lockstep rollout
    ap.add_argument("--beta", type=float, default=0.5,
                    help="fixed inverse temperature; pick so beta*sigma(reward) ~ 1-4")              # beta is reward-scale dependent: see the module docstring
    ap.add_argument("--surrogate", default=None,
                    help="deep-ensemble dir; REQUIRED for a fair comparison with PPO")               # Without this the GFlowNet optimises a different objective than PPO
    ap.add_argument("--w_affinity", type=float, default=0.7)                                         # Match PPO's Stage-2 weight exactly for the affinity component
    ap.add_argument("--w_diversity", type=float, default=0.0)                                        # Must stay 0: the rolling-archive penalty is non-stationary
    ap.add_argument("--warm_start", default=None, help="PPO policy.pt to initialise the trunk")      # Disclosed asymmetry, standard practice to transfer representation
    ap.add_argument("--seed", type=int, default=0)                                                   # Replicate seed ensuring mathematical determinism
    ap.add_argument("--device", default=None)                                                        # Force a device manually, bypassing auto-detection if provided
    ap.add_argument("--max_atoms", type=int, default=25)                                             # Must match the checkpoints exactly to avoid tensor shape mismatches
    a = ap.parse_args()

    # -----------------------------------------------------------------------------------------------
    # Hardware & Environment Initialization: Lock seeds, bind compute devices, parse the target YAML,
    # rebuild shared action space, set the GNN input width and ensure the output directory exists.
    # -----------------------------------------------------------------------------------------------
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)                           # Deterministic replicate locking across PyTorch, NumPy, and Python's random
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")                               # Resolve the device prioritizing GPU acceleration if available
    cfg = yaml.safe_load(open(a.config))                                                             # Load the target config containing domain-specific settings
    spec = ActionSpec(max_atoms=a.max_atoms)                                                         # Rebuild the shared action space to guarantee structural grammar symmetry
    node_dim = len(spec.atom_types) + 3 + 3                                                          # GNN input width, identical to PPO's layout (atom one-hot + hyb + flags)
    os.makedirs(a.out, exist_ok=True)                                                                # Ensure the output directory exists on disk, ignoring errors if present

    # -------------------------------------------------------------------------------------
    # Oracle Surrogate Loading
    # Attach the external affinity scorer, failing open with a loud warning if omitted.
    # -------------------------------------------------------------------------------------
    scorer = None                                                                                    # Optional affinity surrogate placeholder initialization
    if a.surrogate:                                                                                  # Attach it so the GFlowNet optimises PPO's objective
        from surrogate.predict import AffinityScorer                                                 # Imported lazily to keep startup light and prevent unnecessary dependency loading
        scorer = AffinityScorer(a.surrogate, device=str(dev))                                        # The deep ensemble instantiated dynamically onto the target device
    else:                                                                                            # Loudly refuse to let an unmatched run masquerade as comparable
        print("[run_gfn] WARNING: no --surrogate. The GFlowNet will optimise a reward WITHOUT the "
              "affinity term that PPO's reward contains. Any PPO-vs-GFlowNet comparison built on "
              "this run is unfair.")

    # ------------------------------------------------------------------------------------------
    # Architectural Matching (FiLM Conditioning)
    # Warm-starting requires exactly matching the same conditioning as PPO (same pocket vector).
    # ------------------------------------------------------------------------------------------
    # Warm-starting transfers PPO's trunk, which was trained WITH pocket FiLM (pocket_dim=1280), 
    # so the GFlowNet must be built with the same conditioning for the weights to line up.
    pocket_dim = 1280 if a.warm_start else 0                                                         # Match the checkpoint's architecture by reserving tensor width for FiLM features
    pocket_vec = None                                                                                # The fixed ESM-2 pocket embedding initialization
    if a.warm_start:                                                                                 # Only needed when the FiLM path is active (warm starting from PPO)
        pv = cfg.get("esm_pocket_embedding")                                                         # Path from the config pointing to the precomputed NumPy vector
        if pv and os.path.exists(pv):                                                                # Load it when present on the filesystem
            pocket_vec = np.load(pv)                                                                 # Install the same conditioning PPO used, completing the architectural mirror

    # -------------------------------------------------------------------------------------
    # Trajectory Balance Training Execution
    # Kick off the core GFlowNet loop passing all identically mapped environment details.
    # -------------------------------------------------------------------------------------
    # Full provenance line for the log establishing the precise runtime layout
    print(f"[run_gfn] device={dev} num_actions={spec.num_actions} node_dim={node_dim} "
          f"steps={a.steps} batch={a.batch} beta={a.beta} seed={a.seed} "
          f"warm_start={a.warm_start} pocket_dim={pocket_dim}")                                      
    # Train at a FIXED beta (beta_start == beta_end) using the Trajectory Balance algorithm
    train(build_make_env(spec, dev, cfg, scorer, a.w_affinity, a.w_diversity),
          node_dim, spec.num_actions, device=dev, steps=a.steps, batch=a.batch, out=a.out,
          beta_start=a.beta, beta_end=a.beta, pocket_dim=pocket_dim, pocket_vec=pocket_vec,
          warm_start=a.warm_start, seed=a.seed)                                                      


if __name__ == "__main__":
    main()