"""
=======================================
GFlowNet Forward Policy Agent (Stage-3)
=======================================

This script defines the `GFlowNetAgent`, adapting the standard PPO architecture into a 
Generative Flow Network (GFlowNet). A GFlowNet learns to sample terminal molecules with a 
probability exactly proportional to their reward, P(x) ~ R(x). This means it discovers and 
samples many diverse high-reward modes by construction, rather than relying on an explicit 
entropy or diversity penalty like PPO.

Critically, this agent reuses the EXACT same molecule-building MDP, node/edge featurization, 
and GINE-GATv2 GNN trunk as the PPO agent (`gnn_agent.MoleculeAgent`). Therefore, any 
performance or diversity differences between the PPO and GFlowNet baselines are strictly 
attributable to the Trajectory Balance training objective, not the underlying architecture.

The agent introduces two conceptual additions beyond the shared policy head:
1. `log_Z`: A single learnable scalar estimating the log partition function (required for 
   the Trajectory Balance loss - should be eventually equal the sum of all rewards).
2. Backward Policy (P_B): Assumed deterministic (log P_B = 0) in this implementation, since 
   our atom-by-atom builder reaches each molecule via a canonical action ordering. If multiple 
   orderings exist, a uniform backward policy can be swapped in without changing this interface.
"""
import torch
import torch.nn as nn

from gnn_agent import MoleculeAgent


class GFlowNetAgent(nn.Module):
    """
    GFlowNet Agent wrapping the shared GNN architecture for proportional reward sampling.
    
    Inherits from `torch.nn.Module` and instantiates the `MoleculeAgent` as its core policy.
    It ignores the PPO critic (value head) entirely, utilizing only the actor (logits). 
    It registers `log_Z` as a learnable parameter to satisfy the GFlowNet Trajectory Balance 
    equation. It also provides utility functions for pocket conditioning and safe, masked 
    log-probability calculations.
    
    Args:
        node_dim (int): Dimensionality of the input node features.
        num_actions (int): Size of the flat discrete action space.
        hidden_dim (int, optional): Hidden dimensionality for the GNN message passing. Defaults to 128.
        pocket_dim (int, optional): Dimensionality of the target protein pocket conditioning vector. Defaults to 0.
        
    Example:
        >>> agent = GFlowNetAgent(node_dim=12, num_actions=119)
        >>> print(agent.log_Z.item())
        0.0
    """
    def __init__(self, node_dim, num_actions, hidden_dim=128, pocket_dim=0):
        # -----------------------------------------------------------------------------------------
        # Module Initialization
        # Instantiate the shared GNN policy trunk and the learnable partition function parameter.
        # -----------------------------------------------------------------------------------------
        super().__init__()                                                                          # Initialize the base PyTorch nn.Module to properly register sub-modules and parameters
        # GFlowNetAgent reuses the MoleculeAgent architecture for its policy, ignoring the value head.
        self.policy = MoleculeAgent(node_dim, num_actions, hidden_dim=hidden_dim, pocket_dim=pocket_dim) # reuse the full policy trunk + action head; we ignore its value head and use only the logits
        # Learnable log partition function parameter for Trajectory Balance loss; It is a single scalar initialized to zero.
        self.log_Z = nn.Parameter(torch.zeros(1))                                                   # learned log partition function estimating the total reward mass of the environment

    def set_pocket(self, vec):
        """
        Injects the target protein pocket conditioning vector into the underlying policy.
        
        Acts as a passthrough, directly calling the `set_pocket` method on the encapsulated 
        `MoleculeAgent` to enable target-aware molecule generation.
        
        Args:
            vec (torch.Tensor): The computed embedding vector representing the target pocket.
            
        Returns:
            None
        """
        # -----------------------------------------------------------------------------------------
        # Conditioning Passthrough
        # -----------------------------------------------------------------------------------------
        self.policy.set_pocket(vec)                                                                 # Delegate the pocket embedding injection to the internal MoleculeAgent trunk

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Executes a forward pass to compute unnormalized action scores (logits).
        
        Passes the batched graph tensors through the underlying `MoleculeAgent`. It catches 
        both the actor logits and the critic value, but actively discards the critic value 
        as GFlowNets do not utilize standard state-value baselines.
        
        Args:
            x (torch.Tensor): Node features tensor of shape [num_atoms, node_dim].
            edge_index (torch.Tensor): Graph connectivity tensor of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features tensor of shape [num_edges, edge_dim].
            batch (torch.Tensor): Assignment vector mapping nodes to their respective graphs.
            
        Returns:
            torch.Tensor: The unnormalized policy logits of shape [batch_size, num_actions].
        """
        # -----------------------------------------------------------------------------------------
        # Forward Policy Execution
        # Extract actor logits from the GNN and intentionally discard the unused critic value.
        # -----------------------------------------------------------------------------------------
        logits, _ = self.policy(x, edge_index, edge_attr, batch)                                    # [B, A] Evaluate the graph through the shared trunk, keeping logits and dropping the value estimate
        return logits                                                                               # Return the raw, unmasked forward policy action logits

    def masked_log_prob(self, logits, mask, action):
        """
        Computes the log probability of a specific action given the state, log P_F(action | state).
        
        Applies a safe numerical mask to the unnormalized logits, replacing invalid action 
        slots with the smallest representable float for the active dtype. It then computes 
        the log-softmax over the masked logits to get valid probabilities, and finally uses 
        `gather` to extract the exact log probability of the specifically chosen action.
        
        Args:
            logits (torch.Tensor): Raw action scores from the forward pass [batch_size, num_actions].
            mask (torch.Tensor): Boolean mask where True indicates a legally valid action.
            action (torch.Tensor): The specific integer action index actually taken [batch_size].
            
        Returns:
            torch.Tensor: The log probability of the taken action [batch_size].
        """
        # -----------------------------------------------------------------------------------------
        # Masked Probability Calculation: Safely zero-out illegal moves (finfo.min over all invalid
        # logits) and compute the log likelihood of the traversed trajectory.
        # -----------------------------------------------------------------------------------------
        neg = torch.finfo(logits.dtype).min                                                         # Retrieve the absolute smallest representable number for the active tensor float precision
        masked = torch.where(mask.bool(), logits, torch.full_like(logits, neg))                     # Overwrite structurally invalid action logits with the extreme negative safe minimum
        logp = torch.log_softmax(masked, dim=-1)                                                    # Compute the log-softmax across the masked logits to yield a normalized log-probability distribution
        # Return the log probability of the specific action taken for each graph of the batch (a 1D tensor of shape [batch_size]).
        return logp.gather(-1, action.view(-1, 1)).squeeze(-1)                                      # Extract and return strictly the scalar log probability for the singular action that was actually executed