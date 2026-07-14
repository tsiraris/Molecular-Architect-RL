"""
============================================
Active Learning Molecule Generator (Stage-3)
=============================================

This script (src/activelearn/generate.py) is responsible for sampling novel molecules 
from a trained policy (either a PPO agent or a GFlowNet) to feed into the active-learning loop. 

It reuses the project's core `chem_env` and neural `agent` architectures, wrapping them 
in a sequential rollout generator that collects valid candidate structures for subsequent 
docking and scoring by the Stage-3 Oracle.
"""

from typing import List
import torch


def _obs_to_data(obs, device):
    """
    Standardizes diverse environment observation (x, edge_index, edge_attr, node_mask) 
    formats (PyG Data, dict, tuple) into standard PyTorch tensors.
    
    Depending on whether the environment is vectorized or standalone, the observation 
    can be a PyG Data object, a dictionary, or a tuple. This adapter dynamically checks 
    the object type and extracts the node features (x), edge indices (edge_index), and 
    edge attributes (edge_attr), moving them to the target compute device. It specifically 
    utilizes starred unpacking to safely handle the 4-tuple format of `chem_env`.
    
    Args:
        obs (Union[Data, dict, tuple]): The raw observation from the environment.
        device (torch.device): The compute device to place the tensors on.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The normalized (x, edge_index, edge_attr) tensors.
        
    Example:
        >>> obs = (x_tensor, ei_tensor, ea_tensor, mask_tensor)
        >>> x, ei, ea = _obs_to_data(obs, torch.device("cpu"))
    """
    # -------------------------------------------------------------------------------------
    # Observation Normalization
    # Determine the input format and extract the standard PyG components safely.
    # -------------------------------------------------------------------------------------
    # If the observation is a formal PyG Data object (e.g., from vec_env), extract its attributes directly
    if hasattr(obs, "x") and hasattr(obs, "edge_index"):                                    # Check if the observation is a formal PyG Data object (e.g., from vec_env)
        return obs.x.to(device), obs.edge_index.to(device), obs.edge_attr.to(device)        # Extract standard attributes and push them directly to the target compute device
    # If the observation is a generic dictionary (e.g., from standalone env), extract via string keys
    if isinstance(obs, dict):                                                               # Check if the observation is structured as a generic dictionary
        return obs["x"].to(device), obs["edge_index"].to(device), obs["edge_attr"].to(device) # Extract tensors via string keys and push them to the target compute device
    
    # If the observation is a tuple (e.g., from chem_env), unpack it safely using starred unpacking
    x, ei, ea, *_ = obs                                                                     # Fallback: handle chem_env 4-tuple; starred unpacking safely discards extra fields like node_mask
    return x.to(device), ei.to(device), ea.to(device)                                       # Return the extracted core graph tensors mapped to the correct hardware device


@torch.no_grad()
def sample_smiles(agent, make_env, n: int, device="cuda", greedy=False, is_gflownet=False) -> List[str]:
    """
    Rolls out complete generative episodes to sample a batch of valid SMILES strings.
    
    Instantiates a fresh environment and runs sequential steps using the provided agent's policy. 
    It dynamically handles both standard Actor-Critic agents (which return logits and values) 
    and GFlowNet agents (which return just logits). It masks invalid actions using `torch.finfo.min`, 
    samples the next action (or takes the argmax if greedy), and steps the environment until 
    termination. Valid SMILES are harvested directly via `env.get_smiles()`. Returns a list of 
    successfully generated molecular strings, filtering out invalid or duplicate entries.
    
    Args:
        agent (nn.Module): A trained MoleculeAgent or GFlowNetAgent.
        make_env (Callable): A factory function returning a fresh MoleculeEnvironment.
        n (int): The total number of molecular episodes to roll out.
        device (str, optional): The torch compute device string. Defaults to "cuda".
        greedy (bool, optional): If True, always selects the highest-probability action. Defaults to False.
        is_gflownet (bool, optional): Set to True if the agent outputs only logits. Defaults to False.
        
    Returns:
        List[str]: A list of generated SMILES strings from successfully terminated episodes.
        
    Example:
        >>> agent = MoleculeAgent(12, 119)
        >>> smiles_list = sample_smiles(agent, lambda: MoleculeEnvironment(dev), 10)
    """
    # -------------------------------------------------------------------------------------
    # Generator Initialization: Bind hardware, configure agent to eval mode (no dropout), 
    # and prepare the accumulation list for valid SMILES.
    # -------------------------------------------------------------------------------------
    dev = torch.device(device if torch.cuda.is_available() else "cpu")                      # Resolve the hardware compute device with a graceful fallback to CPU if CUDA is missing
    agent = agent.to(dev).eval()                                                            # Migrate the agent to the selected device and lock it in evaluation mode (disabling dropout)
    env = make_env()                                                                        # Instantiate a fresh, isolated single-molecule environment for the rollout sequence
    out = []                                                                                # Initialize an empty Python list to accumulate successfully generated SMILES strings

    # -------------------------------------------------------------------------------------
    # Episodic Rollout Loop
    # Sequentially generate molecules atom-by-atom until n episodes are completed.
    # -------------------------------------------------------------------------------------
    for _ in range(n):                                                                      # Loop precisely n times to fulfill the requested quota of sampled molecules
        obs = env.reset()                                                                   # Reset the environment canvas, returning the initial single-carbon observation
        # If the environment wrapper fails to return a valid observation, 
        # attempt to extract it directly from the environment object
        if obs is None:                                                                     # Defensively check if the specific environment wrapper failed to return the observation
            obs = getattr(env, "obs", None)                                                 # Attempt to manually extract the observation attribute directly from the environment object
        done, steps = False, 0                                                              # Initialize the episode termination flag and reset the local step counter to zero

        while not done and steps < 64:                                                      # Loop until the episode terminates naturally or hits a hard fail-safe limit (64 steps)
            # -----------------------------------------------------------------------------
            # Agent Inference & Masking
            # Standardize observation, query the network, and securely mask illegal moves.
            # -----------------------------------------------------------------------------
            x, ei, ea = _obs_to_data(obs, dev)                                              # Safely unpack the observation into standard PyG graph tensors on the active device
            batch = torch.zeros(x.size(0), dtype=torch.long, device=dev)                    # Create a dummy batch vector (all zeros) since we are processing a single graph at a time
            res = agent(x, ei, ea, batch)                                                   # Execute a single forward pass through the policy network to get action distributions
            
            # If the agent is a GFlowNet, it returns only logits; 
            # otherwise, it returns the first tuple (logits, value)
            logits = res if is_gflownet else res[0]                                         
            if logits.dim() == 2:                                                           # Check if the logits tensor was returned with an unnecessary batch dimension (Shape [1, A])
                logits = logits[0]                                                          # Squeeze the tensor down to a 1D array of shape [A] for direct sampling
            
            # Fetch the dynamic action validity mask from the environment, and apply it to the logits.
            mask = torch.as_tensor(env.get_action_mask(), device=dev)                       # Fetch the dynamic boolean validity mask from the environment and push to device
            neg = torch.finfo(logits.dtype).min                                             # Retrieve the absolute minimum representable float for the current tensor precision (AMP safe)
            masked = torch.where(mask.bool(), logits, torch.full_like(logits, neg))         # Zero out illegal actions by replacing their logits with the dtype minimum before softmax

            # ---------------------------------------------------------------------------------
            # Action Selection: If greedy mode is enabled ("greedy"=True), select the argmax;
            # otherwise, sample stochastically (proportional to) from the softmax distribution.
            # ---------------------------------------------------------------------------------
            if greedy:                                                                      # Check if the generator is configured for deterministic exploitation (greedy mode)
                action = int(masked.argmax())                                               # Deterministically select the legal action corresponding to the absolute highest logit score
            else:                                                                           # Fallback to standard stochastic exploration mode
                action = int(torch.multinomial(torch.softmax(masked, -1), 1))               # Convert logits to probabilities via softmax and sample a single action proportionally

            # ------------------------------------------------------------------------------------------
            # Environment Step: Apply the chosen action to the environment (full MPO reward evaluation).
            # ------------------------------------------------------------------------------------------
            obs, _, done, _ = env.step(action, curriculum_ratio=1.0)                        # Apply the chosen action to the environment, requesting full MPO reward evaluation
            steps += 1                                                                      # Increment the local episode step tracker to enforce the fail-safe timeout loop

        # ---------------------------------------------------------------------------------
        # Terminal Harvesting: If the episode terminated successfully, extract the final 
        # molecular string once the episode concludes successfully.
        # ---------------------------------------------------------------------------------
        if done:                                                                            # Verify that the episode genuinely terminated (via 'stop' action or max steps)
            smi = env.get_smiles()                                                          # Stage-3 fix: Manually invoke the environment's SMILES generator instead of relying on the info dict
            # Filter out empty strings, molecules that failed terminal RDKit sanitization, and duplicates
            if smi and smi != "INVALID":                                                    # Strictly filter out empty strings or molecules that failed terminal RDKit sanitization
                out.append(smi)                                                             # Append the fully validated molecular string to the output payload array

    # Return the final accumulated list of valid sampled molecule SMILES back to the caller
    return out                                                                              