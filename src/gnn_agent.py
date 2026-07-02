"""
=========================================
Graph Neural Network Agent (Actor-Critic)
=========================================

This script defines the `MoleculeAgent`, the "brain" of the reinforcement learning pipeline 
for de novo drug design. Its primary role is to ingest a 2D graph representation of a 
partially built molecule and output both a policy distribution (which chemical action to take next) 
and a value estimate (how good the reward is expected to be for this molecular state).

The architecture is a three-layer message-passing Graph Neural Network (GNN). 
1. It begins with a GINEConv layer to maximize discriminative power over local chemical 
   environments by deeply integrating bond-type edge features.
2. It follows with two layers of GATv2Conv (Graph Attention Network v2) to capture 
   long-range dependencies and route important information across the molecule.
3. Node embeddings are then collapsed into a single global graph embedding via an 
   AttentionalAggregation pool, dynamically weighing the most "interesting" atoms.
4. Finally, this graph vector is routed through twin MLP heads: an Actor (policy logits) 
   and a Critic (state value). 

Note: the forward pass applies a dynamic safety mask using `torch.finfo.min` to 
zero-out chemically invalid actions before softmax/sampling, ensuring numerical 
stability, especially under Automatic Mixed Precision (AMP) training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation


class MoleculeAgent(nn.Module):
    """
    An Actor-Critic Graph Neural Network for step-by-step molecular graph generation.
    
    Inherits from `torch.nn.Module` and defines a shared GNN backbone (GINE + GATv2) 
    that learns topological and chemical representations of an RDKit graph. The shared 
    embedding is branched into an actor head to predict valid structural modifications 
    and a critic head to estimate the expected future multi-parameter objective (MPO) reward.
    
    Args:
        num_node_features (int): Dimensionality of the input node features (e.g., atom type, hybridization etc.)
        num_actions (int): Size of the flat discrete action space (stop, add, focus, ring).
        hidden_dim (int, optional): The base hidden dimensionality for message passing. Defaults to 64.
        edge_dim (int, optional): Dimensionality of the input edge features (e.g., bond type one-hot). Defaults to 4.
        
    Example:
        >>> agent = MoleculeAgent(num_node_features=12, num_actions=119)
        >>> print(agent.actor[2].out_features)
        119
    """
    def __init__(self, num_node_features, num_actions, hidden_dim=64, edge_dim=4, pocket_dim=0):
        # ----------------------------------------------------------------------------------------------------
        # Module Initialization: Call the parent class constructor to register this architecture with PyTorch.
        # ----------------------------------------------------------------------------------------------------
        super(MoleculeAgent, self).__init__()                                                       # Initialize the base nn.Module to properly register parameters and sub-modules

        # -------------------------------------------------------------------------------------------------------------
        # Stage-2 Pocket Conditioning Setup
        # Instantiates the structural FiLM conditioning variables and registers the static pocket buffer.
        # -------------------------------------------------------------------------------------------------------------
        self.pocket_dim = pocket_dim                                                                                # Dimensionality of the target-pocket embedding (0 disables conditioning)
        self.film = None                                                                                            # FiLM module, built below only when pocket_dim > 0
        # Register a persistent-free tensor buffer for the pocket vector to be stored in (not updated by backprop, not saved in the checkpoint)
        self.register_buffer("pocket_vec", torch.zeros(pocket_dim) if pocket_dim > 0 else torch.zeros(0), persistent=False) # Fixed single-target pocket vector, set via set_pocket()
        # If pocket conditioning is enabled (pocket_dim > 0), register the FiLM module
        if pocket_dim > 0:                                                                                          # Evaluate if pocket conditioning is enabled by checking if dimension is positive
            from pocket.conditioning import FiLM                                                                    # Dynamically import the Feature-wise Linear Modulation module for conditioning
            self.film = FiLM(pocket_dim, hidden_dim * 2)
        
        # -----------------------------------------------------------------------------------------
        # GRAPH ENCODER - Layer 1 (GINEConv)
        # Defines the initial graph isomorphism layer. Bond (edge) and atom (node) features are added 
        # between them for all neighbors of each atom, then summed together, added to the atom's own 
        # (node) features, and then passed in an MLP to update.
        # -----------------------------------------------------------------------------------------
        self.gin_mlp = nn.Sequential(                                                               # Instantiate a sequential multi-layer perceptron to act as the GINE node updater
            nn.Linear(num_node_features, hidden_dim),                                               # Map the raw input node features into the network's internal hidden dimensional space
            nn.ReLU(),                                                                              # Apply a non-linear ReLU activation to allow learning of complex internal representations
            nn.Linear(hidden_dim, hidden_dim)                                                       # Project the features again to maintain the hidden_dim shape: [num_nodes, hidden_dim]
        )                                                                                           

        self.conv1 = GINEConv(self.gin_mlp, edge_dim=edge_dim)                                      # Initialize GINEConv to aggregate neighbor info and bond info (linear projection) before updating the center atom

        # --------------------------------------------------------------------------------------------
        # GRAPH ENCODER - Layer 2 & 3 (GATv2Conv)
        # Looks at the complex local chemical environments generated by GINEConv to define dynamic
        # attention layers to capture long-range topological dependencies. GATv2Conv assigns learnable 
        # importance scores to identify how different sub-structures interact across the molecule.
        # --------------------------------------------------------------------------------------------
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=edge_dim)          # Initialize 4-head attention layer (each "looking" for different chemical phenomena) where heads concatenate, resulting in a hidden_dim * 4 output size

        self.conv3 = GATv2Conv(hidden_dim * 4, hidden_dim * 2, heads=2, concat=False, edge_dim=edge_dim) # Initialize refining attention layer, to average the large Layer-2 feature space with 2 heads (concat=False) into a final [num_nodes, hidden_dim * 2] shape
        
        # ---------------------------------------------------------------------------------------------------------
        # GLOBAL POOLING (Attentional Aggregation):
        # Squeezes the variable-sized graph (N nodes) into a fixed-size 1D tensor representing the entire molecule, 
        # weighting nodes by how "interesting" they are. First, it passes every atom's final embedding 
        # [num_nodes, hidden_dim*2] through a small neural network (pool_gate with a Sigmoid) to generate a score 
        # from 0 to 1, and calculates a weighted sum of all atoms based on these scores [batch_size, hidden_dim*2].
        # ---------------------------------------------------------------------------------------------------------
        self.pool_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())                  # Initialize a small gating NN that outputs a [0, 1] importance score for each node embedding
        
        self.pool = AttentionalAggregation(self.pool_gate)                                          # Initialize the pooling layer to compute a weighted sum of nodes, yielding [batch_size, hidden_dim * 2]

        # ------------------------------------------------------------------------------------------------------------
        # ACTOR HEAD ("Decides which should be the next action" - Policy logits)
        # Decodes the global graph embedding into logits over the discrete action space.
        # A linear layer-> ReLU-> linear layer mapping to 119 dimensions (num_actions), producing raw scores (logits).
        # ------------------------------------------------------------------------------------------------------------
        self.actor = nn.Sequential(                                                                 # Initialize the Actor multi-layer perceptron as a sequential block
            nn.Linear(hidden_dim * 2, hidden_dim),                                                  # Down-project the pooled graph embedding into the base hidden dimensionality
            nn.ReLU(),                                                                              # Apply a non-linear ReLU activation function to process the graph-level features
            nn.Linear(hidden_dim, num_actions)                                                      # Map the hidden representation linearly to unnormalized scores (logits) for every possible action
        )                                                                                           

        # --------------------------------------------------------------------------------------------
        # CRITIC HEAD ("Estimates how good the current state is" - Value estimation)
        # Predicts the expected final MPO reward from this state.
        # Decodes the global graph embedding into a single scalar representing expected future reward.
        # --------------------------------------------------------------------------------------------
        self.critic = nn.Sequential(                                                                # Initialize the Critic multi-layer perceptron as a sequential block
            nn.Linear(hidden_dim * 2, hidden_dim),                                                  # Down-project the same pooled graph embedding into the base hidden dimensionality
            nn.ReLU(),                                                                              # Apply a non-linear ReLU activation function for the value estimator
            nn.Linear(hidden_dim, 1)                                                                # Map the hidden representation linearly to a single scalar value estimating the current state's worth
        )                                                                                           

    def set_pocket(self, vec):
        """
        Installs the fixed target-pocket embedding used by FiLM (single-target setup).
        
        Takes a 1-D numerical array or tensor representing the target protein pocket. 
        Validates its length against the configured `pocket_dim`. If conditioning is enabled, 
        it converts the vector to a PyTorch float32 tensor and stores it in the module's 
        registered persistent buffer (`pocket_vec`), allowing the FiLM layer to modulate graph 
        embeddings during the forward pass. If `pocket_dim <= 0`, this acts as a safe no-op.
        
        Args:
            vec (Union[List[float], numpy.ndarray, torch.Tensor]): A 1-D vector containing 
            target-pocket structural or chemical features.
            
        Returns:
            None
            
        Example:
            >>> agent = MoleculeAgent(12, 119, pocket_dim=256)
            >>> target_features = [0.5] * 256
            >>> agent.set_pocket(target_features)
        """
        # ----------------------------------------------------------------------------------------------------
        # Pocket Vector Installation
        # Validates and stores the target condition vector into the model's registered buffer.
        # ----------------------------------------------------------------------------------------------------
        if self.pocket_dim <= 0:                                                                                    # Check if the network was initialized without pocket conditioning (Stage-1 mode)
            return                                                                                                  # Safe no-op if conditioning is disabled, immediately returning without state modification
        import numpy as _np                                                                                         # Import numpy strictly locally to process standard python arrays or iterables
        # If conditioning is enabled, convert the incoming protein pocket vector to a PyTorch tensor
        t = torch.as_tensor(_np.asarray(vec), dtype=torch.float32)                                                  # Cast the incoming numerical collection into a strict PyTorch 32-bit floating point tensor
        # If the vector length matches the architecture's expected dimension size, save the tensor to 
        # the model's buffer, else raise an assertion error
        assert t.numel() == self.pocket_dim, f"pocket vec dim {t.numel()} != pocket_dim {self.pocket_dim}"          # Validate the total element count matches the architecture's expected dimension size
        self.pocket_vec = t

    def forward(self, x, edge_index, edge_attr, batch, action_mask=None):
        """
        Executes a forward pass through the agent to generate policy logits and a value estimate.
        
        Pushes node features, edge indices, and edge features through the 3-layer message 
        passing stack (GINE -> GATv2 -> GATv2) with ReLU activations. The resulting node 
        embeddings are reduced to a single vector per molecule via Attentional Pooling. 
        This vector passes through the Actor to yield action logits and the Critic to yield 
        a state value. If a boolean mask is provided, chemically invalid actions are overwritten 
        with the minimum representable float value to ensure they are never sampled.
        
        Args:
            x (torch.Tensor): Node features tensor of shape [num_atoms, num_node_features].
            edge_index (torch.Tensor): Graph connectivity tensor of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge features tensor of shape [num_edges, edge_dim].
            batch (torch.Tensor): Assignment vector mapping nodes to their respective graphs in a batch.
            action_mask (torch.Tensor, optional): Boolean tensor where True indicates a valid action.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the `action_logits` (shape: [batch_size, num_actions])
            and the `state_value` (shape: [batch_size, 1]).
            
        Example:
            >>> x = torch.randn(5, 12)
            >>> edge_index = torch.tensor([[0, 1], [1, 0]])
            >>> edge_attr = torch.randn(2, 4)
            >>> batch = torch.zeros(5, dtype=torch.long)
            >>> agent = MoleculeAgent(12, 119)
            >>> logits, value = agent(x, edge_index, edge_attr, batch)
        """
        # -----------------------------------------------------------------------------------------
        # Forward Pass - Graph Encoding
        # Sequentially pass the graph tensors through the message passing layers with activations.
        # -----------------------------------------------------------------------------------------
        x = self.conv1(x, edge_index, edge_attr)                                                    # Apply the GINE Convolution layer to update node features using local topological and bond data
        x = F.relu(x)                                                                               # Apply the ReLU activation function to introduce non-linearity after the first convolution
        
        x = self.conv2(x, edge_index, edge_attr)                                                    # Apply the first GATv2 attention layer to dynamically route information across wider neighborhoods
        x = F.relu(x)                                                                               # Apply the ReLU activation function to the concatenated multi-head attention outputs

        x = self.conv3(x, edge_index, edge_attr)                                                    # Apply the second GATv2 layer to refine embeddings, averaging the attention heads together
        x = F.relu(x)                                                                               # Apply the final ReLU activation function to the node-level embeddings

        # -----------------------------------------------------------------------------------------
        # Forward Pass - Readout and Decoders
        # Aggregate the nodes into a graph embedding and pass it through the actor/critic heads.
        # -----------------------------------------------------------------------------------------
        graph_embedding = self.pool(x, batch)                                                       # Compress the node embeddings into a batch-aware global graph embedding using the attention pooler
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Stage-2 FiLM Modulation (Optional) 
        # Before it reaches the actor/critic heads, modulate the graph embedding of each molecule by the (fixed) target-pocket vector via FiLM (output = graph_emb * (1 + gamma) + beta). 
        # No-op when pocket conditioning is disabled, preserving exact Stage-1 behaviour.
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.film is not None and self.pocket_vec.numel() > 0:                                                   # Ensure the FiLM module is initialized and the pocket buffer possesses actual data points
            graph_embedding = self.film(graph_embedding, self.pocket_vec.to(graph_embedding.device))                # Per-feature scale/shift conditioned on the pocket
        
        action_logits = self.actor(graph_embedding)                                                 # Pass the global graph embedding through the Actor MLP to compute raw action selection logits
        state_value = self.critic(graph_embedding)                                                  # Pass the global graph embedding through the Critic MLP to compute the baseline state value

        # -----------------------------------------------------------------------------------------
        # Forward Pass - Safe Action Masking
        # Mask out illegal action moves securely, replacing the outdated -1e9 hack with dtype minimums,
        # to make sure that their softmax probabilty will be equal to zero.
        # -----------------------------------------------------------------------------------------
        if action_mask is not None:                                                                 # Check if a boolean environment validity mask was provided for the current step
            invalid_mask = ~action_mask                                                             # Invert the boolean mask to identify strictly the indices of chemically invalid actions
            
            min_value = torch.finfo(action_logits.dtype).min                                        # Retrieve the absolute smallest representable number for the specific active float precision (e.g., fp16 or fp32)

            action_logits = action_logits.masked_fill(invalid_mask, min_value)                      # Overwrite the logits of invalid actions with the safe minimum to drive their softmax probability to zero
            
        return action_logits, state_value                                                           # Yield the processed policy logits and scalar state value to the training loop