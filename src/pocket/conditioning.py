"""
=======================================================================================================
Stage-2 Pocket Conditioning Module: FiLM conditioning of the graph embedding on a target-pocket vector
=======================================================================================================

This module introduces Feature-wise Linear Modulation (FiLM) to the architecture, 
which lets the same policy gracefully target different protein pockets. The pocket
vector produces a per-feature scale (gamma) and shift (beta) applied to the pooled 
graph embedding just before it is routed to the actor/critic heads. 

For a single target (e.g., KRAS G12C), the pocket vector is constant, so this
effectively acts as a learned target-specific bias. The true value emerges during 
multi-pocket ablation: swapping the pocket vector at evaluation forces the molecule 
distribution to shift, providing the "genuinely target-conditioned" evidence figure.

Default-off Behavior: 
If the agent is built with pocket_dim=0 (or no pocket vector is set), the agent skips
FiLM entirely and behaves byte-for-byte identically to the unconditioned Stage-1 policy.
"""
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) neural network layer for conditioning.
    
    Learns to condition an input graph representation (pooled graph embedding of the molecule)
    on external contextual information (the pocket vector). FiLM applies an affine transformation, 
    projecting the context vector (pocket vector) into scale (gamma) and shift (beta) coefficients, 
    which then affinely modulate the graph embedding of the molecule based on the equation:
                            
                            FiLM(x) = (1 + gamma(c)) * x + beta(c)
    where: 
    - x is the graph embedding of the molecule,
    - c is the context vector (pocket vector), 
    - gamma is the scale parameter whose value is regulated by the pocket vector, 
    - beta is the shift parameter also regulated by the pocket vector.
    This allows the agent's generative policy to adapt based on target geometry 
    (i.e. different ligands with different pockets lead to different gamma and beta values).
    
    Args:
        pocket_dim (int): The dimensionality of the input protein pocket feature vector.
        hid (int): The hidden dimensionality of the incoming graph embedding.
        
    Example:
        >>> film_layer = FiLM(pocket_dim=256, hid=128)
        >>> graph_emb = torch.randn(32, 128)
        >>> pocket_vec = torch.randn(32, 256)
        >>> modulated_emb = film_layer(graph_emb, pocket_vec)
        >>> modulated_emb.shape
        torch.Size([32, 128])
    """
    def __init__(self, pocket_dim: int, hid: int):
        """
        Initializes the FiLM layer with two linear projection networks.
        
        Sets up the linear transformations required to map the context vector to the 
        target hidden dimension. Critically, initializes all parameters to zero so that 
        the initial modulation state is the identity function (no effect). This ensures 
        that the training doesn't violently collapse at the start, but only learns to use 
        the pocket (adjusting Gamma/Beta away from 0) if the pocket actually helps improve 
        the affinity reward.
        
        Args:
            pocket_dim (int): Dimensionality of the target pocket vector.
            hid (int): Dimensionality of the graph embeddings to be modulated.
        """
        # -------------------------------------------------------------------------------------------------------
        # Module Initialization & Linear Projections
        # Register FiLM as an NN module and map the pocket vector into scale (gamma) and shift (beta) parameters.
        # -------------------------------------------------------------------------------------------------------
        super().__init__()                                                                          # Initialize the base nn.Module to properly register network parameters; triggers PyTorch's underlying initialization script to register the FiLM layer inside the neural network ecosystem.
        self.gamma = nn.Linear(pocket_dim, hid)                                                     # Project the pocket vector into a scale (gamma) parameter matching the hidden dim
        self.beta = nn.Linear(pocket_dim, hid)                                                      # Project the pocket vector into a shift (beta) parameter matching the hidden dim
        
        # -----------------------------------------------------------------------------------------
        # Identity Initialization
        # init to identity modulation: gamma->0 so (1+gamma)->1, beta->0, Fout = Fin
        # Ensures the untrained layer does not initially disrupt the graph embeddings.
        # Stage-2 target-conditioned agent begins training mathematically identical to the Stage-1 
        # unconditioned agent, ensuring the training doesn't violently collapse at the start.
        # -----------------------------------------------------------------------------------------
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)                          # Initialize gamma weights and biases to exactly zero to start with no scaling effect
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)                            # Initialize beta weights and biases to exactly zero to start with no shifting effect

    def forward(self, graph_emb: torch.Tensor, pocket_vec: torch.Tensor) -> torch.Tensor:
        """
        Applies the FiLM conditioning modulation to the input graph embeddings.
        
        First guarantees the pocket vector matches the batch dimension of the graph 
        embeddings (handling single 1D pocket vectors via broadcasting). It then 
        computes the gamma and beta shifts and applies the affine transformation: 
        output = graph_emb * (1 + gamma) + beta.
        
        Args:
            graph_emb (torch.Tensor): The pooled graph embedding tensor of shape [B, hid].
            pocket_vec (torch.Tensor): The contextual pocket vector of shape [B, pocket_dim] or [pocket_dim] (broadcast).
            
        Returns:
            torch.Tensor: The dynamically modulated graph embedding of shape [B, hid].
        """
        # --------------------------------------------------------------------------------------------
        # Dimensionality Alignment
        # If the pocket vector is a flat 1D tensor, expand it to match the graph embedding batch size.
        # --------------------------------------------------------------------------------------------
        if pocket_vec.dim() == 1:                                                                   # Check if the provided pocket vector is a flat 1D tensor missing a batch dimension
            pocket_vec = pocket_vec.unsqueeze(0).expand(graph_emb.size(0), -1)                      # Add a batch dimension and broadcast the pocket vector to match the graph batch size
        
        # -----------------------------------------------------------------------------------------
        # Feature-wise Modulation
        # Compute projections and apply the affine transformation to the graph embedding.
        # -----------------------------------------------------------------------------------------
        return graph_emb * (1.0 + self.gamma(pocket_vec)) + self.beta(pocket_vec)                   # Apply the learned affine transformation (scale + shift) to the graph embedding