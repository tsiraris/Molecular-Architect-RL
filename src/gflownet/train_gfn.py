"""
======================================================
Trajectory Balance training for the GFlowNet (Stage 3)
======================================================

This script implements the Trajectory Balance (TB) training objective for the GFlowNet.
For a trajectory tau = s0 -> ... -> sn = x built by the forward policy P_F, 
with a deterministic backward policy (log P_B = 0) and a learned scalar log_Z:

    L_TB(tau) = ( log_Z + sum_t log P_F(s_{t+1}|s_t)  -  beta * reward(x) )^2

Where we use log R(x) = beta * reward(x). This ensures that arbitrary-sign composite 
rewards are handled cleanly. Beta (beta) acts as an inverse temperature parameter, 
which is annealed up over training to sharpen the distribution toward the best molecules. 
At the minimum of this loss, the sampler satisfies: P(x) ~ exp(beta * reward(x)).

The trajectory is generated on the exact same `chem_env` used by PPO. This ensures that 
reward shaping (gate, property, affinity+uncertainty, diversity, warhead) is identical 
between the two algorithms, allowing for rigorous comparison.
"""
import argparse

import torch

from gflownet.agent import GFlowNetAgent


def _obs_to_data(obs, device):
    """
    Converts a raw chem_env observation (x, edge_index, edge_attr) into standard PyTorch Geometric tensors.

    Handles multiple potential observation formats (PyG Data object, dictionary, or tuple) 
    to ensure compatibility. Starred unpacking is used to safely discard extra fields like
    node_mask and prevent 'ValueError: too many values to unpack'.

    Args:
        obs (Any): The raw observation from the environment (Data, dict, or tuple).
        device (torch.device): The compute device to which tensors should be moved.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (node_features, edge_index, edge_attributes) tensors.

    Example:
        >>> x, ei, ea = _obs_to_data(obs_tuple, torch.device('cpu'))
    """
    # --------------------------------------------------------------------------------------------
    # Observation Parsing & Unpacking: Adapt to various observation structures (PyG, Dict, Tuple)
    # and safely extract graph tensors to the target device.
    # --------------------------------------------------------------------------------------------
    if hasattr(obs, "x") and hasattr(obs, "edge_index"):                                    # Check if the observation is a standard PyTorch Geometric Data object
        return obs.x.to(device), obs.edge_index.to(device), obs.edge_attr.to(device)        # Extract attributes directly and cast them to the target compute device
    if isinstance(obs, dict):                                                               # Check if the observation is structured as a Python dictionary
        return obs["x"].to(device), obs["edge_index"].to(device), obs["edge_attr"].to(device) # Access tensors via dictionary keys and cast them to the target compute device
    x, ei, ea, *_ = obs                                                                     # Stage-3 fix #1: chem_env returns 4-tuple; starred unpacking safely absorbs the node_mask and any future fields
    return x.to(device), ei.to(device), ea.to(device)                                       # Cast the unpacked tuple elements to the target compute device and return them


def sample_trajectory(agent, env, device, beta, explore_eps=0.0):
    """
    Rolls out one complete molecular episode to collect a trajectory for GFlowNet training.

    Iteratively steps through the environment using the agent's forward policy P_F 
    (probability arising from action logits) until the molecule is finished or max steps 
    are reached. It masks illegal actions, applies optional epsilon-uniform exploration, 
    and accumulates the log probabilities of the chosen actions (sum_logpf). 
    
    Note: The backward policy is deterministic (log P_B = 0) because the atom-by-atom
    builder reaches each molecule via a canonical, strict action ordering.

    Args:
        agent (GFlowNetAgent): The neural network evaluating state action logits.
        env (MoleculeEnvironment): The environment handling chemical logic and state.
        device (torch.device): The target compute device.
        beta (float): The current inverse temperature scalar for the reward.
        explore_eps (float, optional): Probability of taking a random legal action. Defaults to 0.0.

    Returns:
        Tuple[torch.Tensor, float, Optional[str]]: A tuple containing the accumulated log P_F, 
        the terminal reward scalar, and the generated SMILES string (or None if invalid).

    Example:
        >>> sum_logpf, reward, smiles = sample_trajectory(agent, env, dev, beta=1.5)
    """
    # -----------------------------------------------------------------------------------------
    # Trajectory Initialization
    # Reset the environment and setup accumulators for the Trajectory Balance formula.
    # -----------------------------------------------------------------------------------------
    obs = env.reset()                                                                       # Reset the environment to spawn a fresh episode starting with a single carbon
    if obs is None:                                                                         # Check if the reset failed to return an observation directly
        obs = getattr(env, "obs", None)                                                     # Fallback to extracting the observation from the environment's internal attributes
    sum_logpf = torch.zeros((), device=device)                                              # Initialize a scalar tensor to accumulate log P_F(a|s) across the entire trajectory
    done, reward, steps = False, 0.0, 0                                                     # Initialize loop control flags, reward accumulator, and step counter

    # -----------------------------------------------------------------------------------------
    # Episode Rollout Loop: While the environment is not done and step count is below the hard
    # limit, compute the forward policy logits, and remove the batch dimension for sampling.
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
        # for the Trajectory Balance loss sum_t[log P_F(s_{t+1}|s_t)] term.
        # ------------------------------------------------------------------------------------------
        sum_logpf = sum_logpf + torch.log(probs[action] + 1e-30)                            # Accumulate the log probability of the chosen action for the Trajectory Balance loss, adding a tiny epsilon for numerical stability
        obs, reward, done, _ = env.step(int(action), curriculum_ratio=1.0)                  # Submit the selected action to the environment, forcing curriculum_ratio to 1.0 to get final unscaled rewards
        steps += 1                                                                          # Increment the step tracker to enforce the failsafe timeout

    # -----------------------------------------------------------------------------------------
    # Terminal State Capture: Extract the  SMILES string of the final molecule of the complete
    # trajectory (episode) and ensure invalid states are properly nulled.
    # -----------------------------------------------------------------------------------------
    smiles = env.get_smiles() if done else None                                             # Capture the SMILES string using the explicit getter method only if the episode properly terminated
    if smiles == "INVALID":                                                                 # Check if the environment flagged the terminal graph structure as chemically invalid
        smiles = None                                                                       # Nullify the SMILES string to prevent logging broken data
    # Return the accumulated log P_F, the final reward, and the valid SMILES string (or None if invalid).
    return sum_logpf, float(reward), smiles                                                 


def train(make_env, node_dim, num_actions, device="cuda", steps=3000, batch=16,
          beta_start=1.0, beta_end=4.0, lr=1e-3, hidden_dim=128, pocket_dim=0,
          pocket_vec=None, explore_eps=0.05, out="../artifacts/gfn_kras", log_every=50):
    """
    Trains the GFlowNet agent using the Trajectory Balance (TB) objective.

    Instantiates the GFN agent and configures an optimizer where the partition function (`log_Z`) 
    receives a higher learning rate than the policy trunk. Over `steps` iterations, it anneals 
    the inverse temperature `beta`. In each step, it collects a `batch` of trajectories, 
    calculates the Trajectory Balance loss L_TB = (log_Z + log P_F(tau) - log R(x))^2, 
    accumulates the gradients, and updates the network.

    Args:
        make_env (Callable): Factory function returning a fresh MoleculeEnvironment.
        node_dim (int): Input node-feature dimension (must match chem_env).
        num_actions (int): Total action space size (must match ActionSpec.num_actions).
        device (str, optional): Compute device ("cuda" or "cpu"). Defaults to "cuda".
        steps (int, optional): Total training iterations. Defaults to 3000.
        batch (int, optional): Trajectories collected per gradient step. Defaults to 16.
        beta_start (float, optional): Initial inverse temperature for broad exploration. Defaults to 1.0.
        beta_end (float, optional): Final inverse temperature for focused exploitation. Defaults to 4.0.
        lr (float, optional): Base learning rate for the policy. Defaults to 1e-3.
        hidden_dim (int, optional): GNN hidden dimensionality. Defaults to 128.
        pocket_dim (int, optional): Dimension of the pocket vector (if target-conditioned). Defaults to 0.
        pocket_vec (torch.Tensor, optional): ESM-2 pocket embedding. Defaults to None.
        explore_eps (float, optional): Epsilon-greedy exploration fraction. Defaults to 0.05.
        out (str, optional): Output directory for saving models. Defaults to "../artifacts/gfn_kras".
        log_every (int, optional): Print and log frequency. Defaults to 50.

    Returns:
        GFlowNetAgent: The fully trained agent network.
    """
    # -------------------------------------------------------------------------------------------
    # Training Setup & Initialization: Prepare directories, devices, instantiate a GFlowNet agent
    # object, and initialize the Adam optimizer with two parameter groups: the policy and log_Z.
    # The log_Z parameter group receives a x10 higher learning rate to track the partition 
    # function more quickly (as recommended in the Trajectory Balance paper).
    # -------------------------------------------------------------------------------------------
    import os, json                                                                         # Import os for directory management and json for history serialization
    os.makedirs(out, exist_ok=True)                                                         # Ensure the designated output directory exists, creating it safely if it doesn't
    dev = torch.device(device if torch.cuda.is_available() else "cpu")                      # Resolve the requested device string against actual hardware availability
    agent = GFlowNetAgent(node_dim, num_actions, hidden_dim, pocket_dim).to(dev)            # Initialize the GFlowNet agent with the same trunk as MoleculeAgent and send to device
    if pocket_vec is not None:                                                              # Check if an external target pocket embedding was provided
        agent.set_pocket(pocket_vec)                                                        # Install the ESM-2 pocket vector into the agent for FiLM target conditioning

    # log_Z benefits from a 10x larger LR than the policy (standard TB recommendation).
    opt = torch.optim.Adam([                                                                # Initialize the Adam optimizer with distinct parameter groups
        {"params": [p for n, p in agent.named_parameters() if n != "log_Z"], "lr": lr},     # Group 1: Bind the Policy trunk and action head to the standard base learning rate
        {"params": [agent.log_Z], "lr": lr * 10},                                           # Group 2: Bind the Partition function (log_Z) to a 10x accelerated learning rate to track the normalizer quickly
    ])                                                                                      
    env = make_env()                                                                        # Instantiate a single sequential environment for trajectory generation
    history = []                                                                            # Initialize an empty list to store dictionary rows of training metrics

    # -----------------------------------------------------------------------------------------
    # Main Training Loop
    # Iterate over global steps, annealing beta, collecting batches, and applying gradients.
    # -----------------------------------------------------------------------------------------
    for it in range(steps):                                                                 # Loop continuously through the specified total number of training gradient steps
        # Anneal inverse temperature: low beta early (broad exploration), high beta late (focus on best molecules)
        beta = beta_start + (beta_end - beta_start) * it / max(1, steps - 1)                
        losses, rewards = [], []                                                            # Initialize temporary lists to track metrics for the current batch
        opt.zero_grad()                                                                     # Clear all previously accumulated gradients from the optimizer state
        # For each trajectory in the batch
        for _ in range(batch):                                                              # Collect `batch` number of independent trajectories before executing a gradient update
            # Roll out one full episode to compute accumulated log probability and terminal reward
            sum_logpf, reward, smi = sample_trajectory(agent, env, dev, beta, explore_eps)  # Roll out one full episode to compute accumulated log probability and terminal reward
            # Calculate the log R(x) = beta * reward
            log_r = beta * reward                                                           # Calculate log R(x) = beta * reward (this algebraic trick handles negative composite rewards correctly)
            # Compute the Trajectory Balance loss: L_TB = (log_Z + log P_F(tau) - log R(x))^2
            loss = (agent.log_Z + sum_logpf - log_r) ** 2                                   
            loss.backward()                                                                 # Run backpropagation to accumulate gradients for this specific trajectory into the network parameters
            # Append the detached scalar loss to the batch metric tracker
            losses.append(float(loss.detach()))                                             
            # Append the raw scalar reward to the batch metric tracker
            rewards.append(reward)                                                          

        # Clip gradients to 1.0 and step the optimizer to apply the accumulated batch gradients
        # to the network weights.
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)                             # Clip global gradient norms strictly to 1.0 to prevent explosive instability during updates
        opt.step()                                                                          # Command the optimizer to apply the accumulated batch gradients to the network weights

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

    # Save the fully trained agent weights and the accumulated training history 
    # to disk for future evaluation or resumption.
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
    raise SystemExit("Wire make_env(), node_dim, num_actions from your repo (6 lines), then call train(...). " # Raise a hard exit warning indicating the script requires integration
                     "See the Stage-3 section of the Summary doc for the exact wiring block.") # Direct the user to the documentation for integration instructions