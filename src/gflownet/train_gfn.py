"""
======================================================
Trajectory Balance training for the GFlowNet (Stage 3)
======================================================

This script implements the Trajectory Balance (TB) training objective for the GFlowNet.
For a trajectory tau = s0 -> ... -> sn = x built by the forward policy P_F, 
with a deterministic backward policy (log P_B = 0) and a learned scalar log_Z:

    L_TB(tau) = ( log_Z + sum_t log P_F(s_{t+1}|s_t)  -  beta * reward(x) )^2

where we use log R(x) = beta * reward(x) so that arbitrary-sign composite rewards are 
handled cleanly and beta is an inverse temperature. At the minimum of this loss, the 
sampler satisfies: P(x) ~ exp(beta * reward(x)).

The trajectory is generated on the SAME chem_env used by PPO, so reward shaping (gate, property,
affinity+uncertainty, diversity, warhead) is identical between the two algorithms.
"""
import argparse
import math
import random

import torch
import numpy as np
from torch_geometric.data import Batch, Data
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")                                                              # Mute RDKit's benign valence/kekulize chatter: earlier runs wrote 14M log lines, which is real I/O time

from gflownet.agent import GFlowNetAgent


def _obs_to_data(obs, device):
    """
    Converts a raw chem_env observation (x, edge_index, edge_attr) into standard PyTorch Geometric tensors.

    Handles multiple potential observation formats (PyG Data object, dictionary, or tuple) 
    to ensure compatibility. Starred unpacking is used to safely discard extra fields (like node_mask) 
    and prevent 'ValueError: too many values to unpack'.

    Args:
        obs (Any): The raw observation from the environment (Data, dict, or tuple).
        device (torch.device): The compute device to which tensors should be moved.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (node_features, edge_index, edge_attributes) tensors.

    Example:
        >>> x, ei, ea = _obs_to_data(obs_tuple, torch.device('cuda'))
    """
    # --------------------------------------------------------------------------------------------
    # Observation Parsing & Unpacking: Adapt to various observation structures (PyG, Dict, Tuple)
    # and safely extract graph tensors to the target device.
    # --------------------------------------------------------------------------------------------
    if hasattr(obs, "x") and hasattr(obs, "edge_index"):                                    # Check if the observation is a standard PyTorch Geometric Data object
        return obs.x.to(device), obs.edge_index.to(device), obs.edge_attr.to(device)        # Extract attributes directly and cast them to the target compute device
    if isinstance(obs, dict):                                                               # Check if the observation is structured as a Python dictionary
        return obs["x"].to(device), obs["edge_index"].to(device), obs["edge_attr"].to(device) # Access tensors via dictionary keys and cast them to the target compute device
    x, ei, ea, *_ = obs                                                                     # Stage-3 fix: chem_env returns 4-tuple; starred unpacking handles any length safely
    return x.to(device), ei.to(device), ea.to(device)                                       # Cast the unpacked tuple elements to the target compute device and return them


def _n_parents(env):
    """
    Counts the DAG in-edges (structural parents) of the current environment state.

    To support a uniform backward policy, this computes how many valid prior states could 
    have resulted in the current molecule/focus state. In this construction MDP, a state's 
    parents are defined by:
      1. Focus shifts: Any other atom could have shifted focus to the current one (n_atoms - 1).
      2. Atom additions: Valid only if the focused atom is a leaf (degree == 1).
      3. Ring closures: Reversible ring bonds incident to the focused atom.
    Summing these provides the normalizer for the uniform backward policy P_B.

    Args:
        env (MoleculeEnvironment): The environment whose `current_mol` and `focus_node_idx` define the state.

    Returns:
        int: The number of structural parents (always >= 1).

    Example:
        >>> parent_count = _n_parents(env)
        >>> print(parent_count)
        2
    """
    # --------------------------------------------------------------------------------------------
    # Backward Policy (P_B) Normalizer Calculation
    # Determine the number of valid preceding states to calculate the uniform backward probability.
    # --------------------------------------------------------------------------------------------
    mol = getattr(env, "current_mol", None)                                                 # Retrieve the current RDKit molecule object from the environment state
    if mol is None:                                                                         # Defensive check: if no molecule exists, default to a single parent origin
        return 1                                                                            # Return 1 to prevent zero-division in logarithmic calculations
    n = mol.GetNumAtoms()                                                                   # Query the current total number of atoms in the molecule
    if n <= 1:                                                                              # Check if the molecule is just the initial single-atom seed state
        return 1                                                                            # The initial state strictly has no real structural parents, return 1
    # Extract the current focus atom
    f = int(getattr(env, "focus_node_idx", 0))                                              # Retrieve the integer index of the currently focused atom
    f = min(f, n - 1)                                                                       # Defensively clamp the focus index to ensure it falls within valid atom bounds
    atom = mol.GetAtomWithIdx(f)                                                            # Extract the specific RDKit atom object currently under focus
    # Number of focus parent states: any of the other existing atoms could have shifted focus here.
    focus_parents = n - 1                                                                   
    # Count add-parents: if the atom is a terminal leaf, it could be the recently added atom.
    add_parent = 1 if atom.GetDegree() == 1 else 0                                          
    # Count ring-parents: tally all reversible ring-closure bonds incident to this focus atom
    ring_parents = sum(1 for b in atom.GetBonds() if b.IsInRing())                          
    return max(1, focus_parents + add_parent + ring_parents)                                # Sum the mutually exclusive parent sources and return (clamped to a minimum of 1)


def sample_trajectory(agent, env, device, beta, explore_eps=0.0):
    """
    Rolls out one complete molecular episode to collect a trajectory for GFlowNet training.

    Iteratively steps through the environment using the agent's forward policy P_F, 
    until the molecule is finished or max steps are reached.
    It accumulates the log probabilities of the chosen actions (sum_logpf) and computes 
    the uniform backward policy (sum_logpb) via `_n_parents` (used downstream to cancel 
    out trajectory length biases in the Trajectory Balance loss).

    Args:
        agent (GFlowNetAgent): The neural network evaluating state action logits.
        env (MoleculeEnvironment): The environment handling chemical logic and state.
        device (torch.device): The target compute device.
        beta (float): The fixed inverse temperature scalar for the reward.
        explore_eps (float, optional): Probability of a random legal action. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, float, float, Optional[str]]: A tuple containing the accumulated 
        log P_F, the accumulated log P_B, the terminal reward scalar, and the generated SMILES.

    Example:
        >>> logpf, logpb, rew, smi = sample_trajectory(agent, env, dev, beta=1.0)
    """
    # -----------------------------------------------------------------------------------------
    # Trajectory Initialization
    # Reset the environment and setup accumulators for both forward and backward policies.
    # -----------------------------------------------------------------------------------------
    obs = env.reset()                                                                       # Start a fresh episode
    if obs is None:                                                                         # Check if the reset failed to return an observation directly
        obs = getattr(env, "obs", None)                                                     # Fallback to extracting the observation from the environment's internal attributes
    sum_logpf = torch.zeros((), device=device)                                              # Accumulate log P_F(a|s) across the trajectory
    sum_logpb = 0.0                                                                         # Accumulate log P_B(s|s') = -log n_parents(s') (uniform backward; cancels length term)
    done, reward, steps = False, 0.0, 0                                                     # Track episode state

    # -----------------------------------------------------------------------------------------
    # Episode Rollout Loop
    # Calculate masked forward probabilities and accumulate structural backward probabilities.
    # -----------------------------------------------------------------------------------------
    while not done and steps < 64:                                                          # Loop until the environment signals termination, with a hard failsafe at 64 steps (env natively stops at 40)
        x, ei, ea = _obs_to_data(obs, device)                                               # Unpack the current 4-tuple observation into graph tensors (incorporating Stage-3 fix #1)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)                     # Generate a zeroed batch index tensor mapping all nodes to a single graph
        logits = agent(x, ei, ea, batch)                                                    # Compute the forward pass once and cache the result
        if logits.dim() == 2:                                                               # Check if the network output retains a batch dimension (shape [1, num_actions])
            logits = logits[0]                                                              # Squeeze the tensor down to a 1D vector of shape [num_actions] for sampling

        # -------------------------------------------------------------------------------------------
        # Action Masking & Forward Policy (P_F) Construction
        # Zero out illegal chemical moves safely before converting logits to probabilities (softmax).
        # -------------------------------------------------------------------------------------------
        mask = env.get_action_mask()                                                        # Retrieve the boolean mask array defining strictly legal chemical actions for this state
        mask = torch.as_tensor(mask, device=device)                                         # Cast the boolean mask to a PyTorch tensor on the active compute device
        neg = torch.finfo(logits.dtype).min                                                 # Retrieve the minimum representable float value as an AMP-safe masking constant
        masked = torch.where(mask.bool(), logits, torch.full_like(logits, neg))             # Overwrite illegal action logits with the minimum value to drive their probability to zero
        probs = torch.softmax(masked, dim=-1)                                               # Apply softmax to the masked logits to create the forward policy probability distribution P_F(.|s)

        # --------------------------------------------------------------------------------------------
        # Action Sampling & Environment Stepping: Every random steps, if exploration is enabled, 
        # select a purely uniform random legal action, and otherwise sample from the policy as normal
        # (the probability of each action is proportional to its forward policy).
        # --------------------------------------------------------------------------------------------
        if explore_eps > 0 and torch.rand(()) < explore_eps:                                # Evaluate the epsilon threshold to decide if a random exploratory action should be taken
            legal = mask.nonzero().flatten()                                                # Extract the flat 1D indices of all currently legal actions from the mask
            action = legal[torch.randint(len(legal), ())]                                   # Sample a purely uniform random action exclusively from the legal action set
        else:                                                                               # If exploration fails or is disabled, default to policy-driven sampling
            action = torch.multinomial(probs, 1).squeeze()                                  # Sample exactly one action index based on the calculated forward policy probabilities
        # ------------------------------------------------------------------------------------------
        # Trajectory Balance Loss Accumulation: Accumulate the log probability of the chosen action
        # (by adding the log P_F(s_{t+1}|s_t) terms), and of the uniform backward policy 
        # (by adding the log P_B(s_t|s_{t+1}) = -log n_parents(s_{t+1}) terms) for the TB loss. 
        # ------------------------------------------------------------------------------------------
        sum_logpf = sum_logpf + torch.log(probs[action] + 1e-30)                            # Accumulate the log probability of the chosen action for the Trajectory Balance loss, adding a tiny epsilon for numerical stability
        a_int = int(action)                                                                 # Decode action index (0 == stop, the deterministic exit)
        obs, reward, done, _ = env.step(int(action), curriculum_ratio=1.0)                  # Submit the selected action to the environment, forcing curriculum_ratio to 1.0 to get final unscaled rewards
        if a_int != 0:                                                                      # Exit action has deterministic backward (log P_B = 0); all others use uniform P_B
            import math as _math                                                            # Local import to avoid touching module header
            sum_logpb += -_math.log(_n_parents(env))                                        # log P_B(s_t|s_{t+1}) = -log n_parents(s_{t+1})
        steps += 1                                                                          # Increment the step tracker to enforce the failsafe timeout

    # -----------------------------------------------------------------------------------------
    # Terminal State Capture: Extract the  SMILES string of the final molecule of the complete
    # trajectory (episode) and ensure invalid states are properly nulled.
    # -----------------------------------------------------------------------------------------
    smiles = env.get_smiles() if done else None                                             # Capture the SMILES string using the explicit getter method only if the episode properly terminated
    if smiles == "INVALID":                                                                 # Check if the environment flagged the terminal graph structure as chemically invalid
        smiles = None                                                                       # Nullify the SMILES string to prevent logging broken data
    # Return thς Σlog P_F, Σlog P_B , the final reward, and the valid SMILES string (or None if invalid).
    return sum_logpf, float(sum_logpb), float(reward), smiles                               # Return the four elements required for TB loss and logging


def sample_batch(agent, envs, device, beta, explore_eps=0.0):
    """
    Rolls out len(envs) complete trajectories IN LOCKSTEP, batching every GNN forward pass.

    This is a pure throughput optimization. Instead of one environment at a time, it evaluates 
    all currently active environments in a single batched graph forward pass (`Batch.from_data_list`).
    Environments that finish early drop out of the active set. The mathematical outcome is 
    identical to sequential rollouts, but GPU utilization is massively increased.

    Args:
        agent (GFlowNetAgent): The active policy network.
        envs (List[MoleculeEnvironment]): Independent environments, one per trajectory.
        device (torch.device): Compute device.
        beta (float): Fixed inverse temperature (unused in generation, kept for signature parity).
        explore_eps (float): Epsilon-uniform exploration probability over legal actions.

    Returns:
        Tuple[List[Tensor], List[float], List[float], List[Optional[str]]]:
        (sum_logPF, sum_logPB, terminal_reward, SMILES) parallel arrays for the batch.

    Example:
        >>> pf, pb, rews, smis = sample_batch(agent, env_list, dev, beta=1.0)
    """
    # -----------------------------------------------------------------------------------------
    # Lockstep Batch Initialization
    # Set up parallel accumulators and trackers for every environment in the array.
    # -----------------------------------------------------------------------------------------
    n = len(envs)                                                                           # Trajectories rolled out in parallel == the gradient-step batch size
    obs = [e.reset() for e in envs]                                                         # Fresh episode in every environment
    sum_logpf = [torch.zeros((), device=device) for _ in range(n)]                          # Per-trajectory accumulator for log P_F (kept on the autograd graph)
    sum_logpb = [0.0] * n                                                                   # Per-trajectory accumulator for log P_B (a constant: uniform backward policy)
    rewards = [0.0] * n                                                                     # Terminal composite reward per trajectory
    smiles = [None] * n                                                                     # Terminal SMILES per trajectory
    active = list(range(n))                                                                 # Indices of environments still running
    steps = 0                                                                               # Guard counter (env terminates at max_steps; 64 is a hard fail-safe)

    # -----------------------------------------------------------------------------------------
    # Batched Rollout Loop
    # Consolidate all active environment states into a single PyG forward pass.
    # -----------------------------------------------------------------------------------------
    while active and steps < 64:                                                            # Step every live environment in lockstep until all have terminated
        data_list = []                                                                      # PyG Data objects for the live environments
        for i in active:                                                                    # Build one graph per live environment
            x, ei, ea = _obs_to_data(obs[i], device)                                        # Reuse the proven 4-tuple -> tensors converter
            data_list.append(Data(x=x, edge_index=ei, edge_attr=ea))                        # Wrap as Data so PyG can batch it
        b = Batch.from_data_list(data_list)                                                 # ONE batched graph for all live environments
        logits = agent(b.x, b.edge_index, b.edge_attr, b.batch)                             # ONE forward pass replaces len(active) separate launches
        if logits.dim() == 1:                                                               # Defensive: squeeze/expand to [n_active, A]
            logits = logits.unsqueeze(0)                                                    # Restore the batch dimension

        masks = torch.stack([torch.as_tensor(envs[i].get_action_mask(), device=device)
                             for i in active])                                              # [n_active, A] boolean legality masks
        neg = torch.finfo(logits.dtype).min                                                 # AMP-safe masking constant
        masked = torch.where(masks.bool(), logits, torch.full_like(logits, neg))            # Mask illegal actions before the softmax
        probs = torch.softmax(masked, dim=-1)                                               # Forward policy P_F(.|s) for every live environment
        acts = torch.multinomial(probs, 1).squeeze(-1)                                      # Sample one action per live environment

        finished = []                                                                       # Environments that terminate on this step
        # For every live environment: sample an action, apply it, and update accumulators
        for k, i in enumerate(active):                                                      # Apply each sampled action to its own environment
            a = int(acts[k])                                                                # The on-policy action
            if explore_eps > 0 and random.random() < explore_eps:                           # Epsilon-uniform exploration over the legal set
                legal = masks[k].nonzero().flatten()                                        # Indices of legal actions for this environment
                if len(legal):                                                              # Guard against a fully-masked state
                    a = int(legal[torch.randint(len(legal), ())])                           # Replace with a uniform legal action
            sum_logpf[i] = sum_logpf[i] + torch.log(probs[k, a] + 1e-30)                    # Accumulate log P_F(a|s) on the autograd graph
            o, rew, done, _ = envs[i].step(a, curriculum_ratio=1.0)                         # Advance this environment
            obs[i] = o                                                                      # Cache the next observation
            if a != 0:                                                                      # Action 0 is the deterministic stop/exit: log P_B = 0 there
                sum_logpb[i] += -math.log(_n_parents(envs[i]))                              # Uniform backward policy: log P_B(s_t|s_{t+1}) = -log n_parents(s_{t+1})
            if done:                                                                        # Episode complete
                rewards[i] = float(rew)                                                     # Capture the terminal composite reward
                sm = envs[i].get_smiles()                                                   # Harvest the terminal molecule
                smiles[i] = None if sm in (None, "INVALID") else sm                         # Drop chemically invalid terminals
                finished.append(i)                                                          # Retire this environment from the live set
        active = [i for i in active if i not in finished]                                   # Shrink the batch as trajectories complete
        steps += 1                                                                          # Advance the lockstep counter

    return sum_logpf, sum_logpb, rewards, smiles                                            # Everything the TB loss needs, for the whole batch


def train(make_env, node_dim, num_actions, device="cuda", steps=3000, batch=16,
          beta_start=1.0, beta_end=4.0, lr=1e-3, hidden_dim=128, pocket_dim=0,
          pocket_vec=None, explore_eps=0.05, out="../artifacts/gfn_kras", log_every=50,
          warm_start=None, logz_lr=1e-1, seed=0):
    """
    Trains the GFlowNet agent using the advanced batched Trajectory Balance (TB) objective.

    Initializes the network (optionally warm-started from a PPO checkpoint to skip cold exploration).
    It configures dual learning rates, heavily accelerating `log_Z` to quickly track the partition function. 
    It maintains a fixed inverse temperature (`beta`) to prevent `log_Z` from chasing a moving target. 
    Over `steps` iterations, it lockstep-samples a `batch` of trajectories, computes the 
    full TB loss incorporating the backward policy (P_B) and a baseline reward shift, and applies gradients.

    Args:
        make_env (Callable): Factory function returning a fresh MoleculeEnvironment.
        node_dim (int): Input node-feature dimension (must match chem_env).
        num_actions (int): Total action space size (must match ActionSpec.num_actions).
        device (str, optional): Compute device string. Defaults to "cuda".
        steps (int, optional): Total training iterations. Defaults to 3000.
        batch (int, optional): Trajectories collected per gradient step. Defaults to 16.
        beta_start (float, optional): Fixed inverse temperature for TB scaling. Defaults to 1.0.
        beta_end (float, optional): Legacy param, unused due to fixed beta requirement. Defaults to 4.0.
        lr (float, optional): Base learning rate for the policy trunk. Defaults to 1e-3.
        hidden_dim (int, optional): GNN hidden dimensionality. Defaults to 128.
        pocket_dim (int, optional): Dimension of target-conditioning vector. Defaults to 0.
        pocket_vec (torch.Tensor, optional): ESM-2 target pocket embedding. Defaults to None.
        explore_eps (float, optional): Epsilon-greedy exploration fraction. Defaults to 0.05.
        out (str, optional): Output directory for artifacts. Defaults to "../artifacts/gfn_kras".
        log_every (int, optional): Evaluation print frequency. Defaults to 50.
        warm_start (str, optional): Filepath to a PPO `state_dict` to jumpstart policy. Defaults to None.
        logz_lr (float, optional): Highly accelerated learning rate for `log_Z`. Defaults to 1e-1.
        seed (int, optional): Global random seed for reproducible runs. Defaults to 0.

    Returns:
        GFlowNetAgent: The fully trained agent network.
    """
    # -------------------------------------------------------------------------------------------
    # Training Setup: Seed enforcement, threading constraints, and agent instantiation.
    # -------------------------------------------------------------------------------------------
    import os, json                                                                         # Import os for directory management and json for history serialization
    os.makedirs(out, exist_ok=True)                                                         # Ensure the designated output directory exists, creating it safely if it doesn't
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)                        # Deterministic seeding so multi-seed replicates are reproducible
    torch.set_num_threads(1)                                                                # One thread per process: several beta/seed runs share the box, and thread thrash costs more than it wins
    dev = torch.device(device if torch.cuda.is_available() else "cpu")                      # Resolve the requested device string against actual hardware availability
    agent = GFlowNetAgent(node_dim, num_actions, hidden_dim, pocket_dim).to(dev)            # Initialize the GFlowNet agent with the same trunk as MoleculeAgent
    if pocket_vec is not None:                                                              # Check if an external target pocket embedding was provided
        agent.set_pocket(pocket_vec)                                                        # Install the ESM-2 pocket vector for FiLM conditioning (optional)

    # ---------------------------------------------------------------------------------------------
    # Warm-Start Transfer: Initialise the GFlowNet trunk + actor (+ FiLM) weights from a trained
    # PPO policy so training begins in a region that already produces valid, reasonable molecules
    # (a cold start on this multi-step construction MDP explores mostly-invalid space and struggles
    # to find reward). The PPO checkpoint is a MoleculeAgent state_dict with identical keys to 
    # agent.policy; we shape-filter so the deeper Stage-3 critic (unused by GFlowNet) is skipped.
    # ---------------------------------------------------------------------------------------------
    if warm_start is not None:                                                              # Check if a path to a pre-trained PPO model checkpoint was provided
        _sd = torch.load(warm_start, map_location="cpu")                                    # Load the PPO checkpoint into CPU memory safely
        _sd = _sd.get("model", _sd) if isinstance(_sd, dict) else _sd                       # Unwrap {"model": ...} if present
        _mine = agent.policy.state_dict()                                                   # GFlowNet's trunk/heads (a MoleculeAgent)
        _compat = {k: v for k, v in _sd.items() if k in _mine and _mine[k].shape == v.shape}  # Keep only shape-matching tensors
        agent.policy.load_state_dict(_compat, strict=False)                                 # Transfer trunk + actor (+ FiLM) weights, ignoring missing keys like the critic
        print(f"[warm-start] loaded {len(_compat)}/{len(_sd)} PPO tensors into the GFlowNet trunk " # Print a status update confirming the successful transfer
              f"(skipped {len(_sd) - len(_compat)}: critic/shape mismatches)")              # Document the number of unused weights (typically the PPO value head)

    # --------------------------------------------------------------------------------------------
    # Optimizer & Log_Z Initialization: Decouple the normalizer's learning rate to allow rapid 
    # scaling to large targets. log_Z benefits from a 10x larger LR than the policy (standard TB).
    # --------------------------------------------------------------------------------------------
    opt = torch.optim.Adam([                                                                # Initialize the Adam optimizer with distinct parameter groups
        {"params": [p for n, p in agent.named_parameters() if n != "log_Z"], "lr": lr},     # Policy trunk + action head
        {"params": [agent.log_Z], "lr": logz_lr},                                           # log_Z needs a fast, high LR: its target is large (~ -E[logPF]+E[logPB]+log_r ~ +20 here); lr*3 was far too slow, so it never converged and the reward signal never reached the policy
    ])
    envs = [make_env() for _ in range(batch)]                                               # One environment per trajectory: the batch is now rolled out in lockstep, not sequentially

    # log_Z initialization: start at 0 and let its fast learning rate (logz_lr, default 1e-1) 
    # find the fixed point on its own rather than hand-setting it. 
    agent.log_Z.data.fill_(0.0)                                                             

    history = []                                                                            # Initialize an empty list to store dictionary rows of training metrics

    # -----------------------------------------------------------------------------------------
    # Main Training Loop
    # Lockstep sample generation, TB residual formulation, and batched backpropagation.
    # -----------------------------------------------------------------------------------------
    for it in range(steps):                                                                 # Loop continuously through the specified total number of training gradient steps
        # Beta must be constant during training. Annealing it makes log_Z chase a moving 
        # target (log_Z tracked beta linearly and absorbed the entire loss -> no learning).
        beta = beta_start                                                                   # Hold inverse temperature fixed for a stable partition function
        opt.zero_grad()                                                                     # Clear gradients at the start of each gradient step

        sum_logpf, sum_logpb, rewards, _smis = sample_batch(agent, envs, dev, beta, explore_eps)  # Roll out the whole batch in lockstep with batched forwards
        
        # Trajectory Balance, summed over the batch: By summing the batch's squared TB 
        # residuals here and calling backward once, reproduces that gradient exactly while
        # replacing `batch` autograd traversals with one.
        R_BASELINE = -2.0                                                                   # Below the reward floor so log_r stays finite and positive; a constant shift is absorbed by log_Z
        total = torch.zeros((), device=dev)                                                 # To acumulate the batch's squared TB residuals
        losses = []                                                                         # Per-trajectory loss values, for logging only
        # For each complete trajectory in the batch, compute its squared TB residual,
        # add it to the total, and backpropagate the accumulated gradient.
        for i in range(len(rewards)):                                                       # One squared residual per complete trajectory
            log_r = beta * (rewards[i] - R_BASELINE)                                        # Proportional target: P(x) ~ exp(beta * reward)
            li = (agent.log_Z + sum_logpf[i] - sum_logpb[i] - log_r) ** 2                   # Full TB residual: (log_Z + sum log P_F - sum log P_B - log R)^2
            total = total + li.squeeze()                                                    # Sum, matching the original per-trajectory backward accumulation
            losses.append(float(li.detach()))                                               # Record for the log row
        total.backward()                                                                    # One backward pass for the entire batch

        # Clip only policy grads, because log_Z must move freely to its large target
        torch.nn.utils.clip_grad_norm_([p for n, p in agent.named_parameters() if n != "log_Z"], 1.0)  
        opt.step()                                                                          # Apply the accumulated gradients

        # -----------------------------------------------------------------------------------------
        # Logging & Checkpointing
        # Periodically emit training statistics and save final artifacts to disk.
        # -----------------------------------------------------------------------------------------
        if it % log_every == 0:                                                             # Check if the current step aligns with the requested logging interval
            row = {"step": it, "beta": round(beta, 3),                                      # Construct a dictionary containing the current step index and active beta value
                   "loss": round(sum(losses) / len(losses), 4),                             # Calculate and format the mean Trajectory Balance loss for the current batch
                   "reward_mean": round(sum(rewards) / len(rewards), 4),                    # Calculate and format the mean terminal reward achieved in the current batch
                   "log_Z": float(agent.log_Z)}                                             # Extract the current estimated partition function scalar directly from the agent
            history.append(row)                                                             # Append the fully populated metric dictionary to the global history list
            print(row)                                                                      # Echo the metric dictionary to the console standard output

    # -----------------------------------------------------------------------------------------
    # Training Teardown: Save the fully trained agent weights and the accumulated training 
    # history to disk for future evaluation or resumption.
    # -----------------------------------------------------------------------------------------
    torch.save(agent.state_dict(), os.path.join(out, "gfn_agent.pt"))                       # Serialize and save the fully trained agent weights to the designated output folder
    json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=2)              # Dump the accumulated JSON history log to a formatted text file
    print(f"[gfn] saved -> {out}")                                                          # Print a final confirmation message indicating successful disk persistence
    return agent                                                                            # Return the trained agent object to the caller


if __name__ == "__main__":
    ap = argparse.ArgumentParser()                                                          # Initialize the command-line argument parser for direct script execution
    ap.add_argument("--out", default="../artifacts/gfn_kras")                               # Define the target output directory flag, defaulting to the artifacts folder
    ap.add_argument("--steps", type=int, default=3000)                                      # Define the total training steps flag, defaulting to 3000 iterations
    ap.add_argument("--device", default="cuda")                                             # Define the compute device flag, defaulting to GPU execution
    args = ap.parse_args()                                                                  # Parse all provided command-line arguments into the args namespace
    raise SystemExit("Wire make_env(), node_dim, num_actions from your repo (6 lines), then call train(...). " ) # Halt direct execution and redirect to the properly wired stage-3 driver script