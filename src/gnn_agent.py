import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation #

class MoleculeAgent(nn.Module):
    def __init__(self, num_node_features, num_actions, hidden_dim=64, edge_dim=4):
        super(MoleculeAgent, self).__init__()
        
        # 1. GRAPH ENCODER
        
        # Layer 1: GINEConv (Graph Isomorphism Network with Edges) - Captures structure (local chemical environments) for maximal expressiveness
        # An MLP able to learn arbitrarily complex rules about how to combine an atom with its neighbors.
        self.gin_mlp = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim) # output shape: [num_nodes, hidden_dim]
        ) 
        
        # Before the neighbor's info is added to the center atom, the bond info is added (linear projection) to the neighbor
        self.conv1 = GINEConv(self.gin_mlp, edge_dim=edge_dim) 

        # Layer 2: GATv2 (Graph Attention Network v2) - Captures long-range dependencies and importance (which neighbors are more important for the current task)
        # Assigns a learnable score (attention weight​) to every bond, deciding how much information to "absorb" from that neighbor for each pairing of nodes
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=edge_dim) # num_heads independent attention "committees", leading to a hidden_dim * num_heads col size (concat=True)
        
        # Layer 3: GATv2 (Refining)
        self.conv3 = GATv2Conv(hidden_dim * 4, hidden_dim * 2, heads=2, concat=False, edge_dim=edge_dim) # Final graph embedding - [num_nodes,hidden_dim * 2] (concat=False means: average the heads instead of concatenating)
        
        # 2. GLOBAL POOLING (Attention-based)
        # Pool gate: Small NN  that looks at each node's final embedding and outputs a score (0 to 1) that determines how much that node should contribute to the final graph embedding (how "interesting" it is).
        self.pool_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        
        # AttentionalAggregation uses these scores to weight the nodes when summing them up into a single graph embedding vector (global representation of the molecule). 
        # This allows the model to focus on the most relevant parts of the molecule for decision-making.
        self.pool = AttentionalAggregation(self.pool_gate) # Output shape: [batch_size, hidden_dim * 2]

        # 3. ACTOR HEAD
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # 4. CRITIC HEAD
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch, action_mask=None):
        
        # --- Layer 1: GINE (Local Chemistry) ---
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # --- Layer 2: GATv2 (Attention) ---
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        # --- Layer 3: GATv2 (Deep Reasoning) ---
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        # --- Readout: Global Attention Pooling ---
        graph_embedding = self.pool(x, batch) 

        # --- Heads ---
        action_logits = self.actor(graph_embedding)
        state_value = self.critic(graph_embedding)

        # --- Safe Masking ---
        if action_mask is not None:
            # 1. Invert mask (True where invalid)
            invalid_mask = ~action_mask
            
            # 2. Get the smallest representable number for the current dtype (float16 or float32)
            min_value = torch.finfo(action_logits.dtype).min

            # 3. Fill with that safe minimum instead of -1e9
            action_logits = action_logits.masked_fill(invalid_mask, min_value)
            
        return action_logits, state_value