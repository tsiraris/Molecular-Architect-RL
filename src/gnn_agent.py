import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation

class MoleculeAgent(nn.Module):
    """
    Actor-Critic GNN Agent.
    
    Structure:
    1. GINEConv: Captures local chemical bonds (Hard Logic).
    2. GATv2Conv: Captures long-range dependencies (Attention).
    3. Global Pooling: Attention-based aggregation (Pharmacophore detection).
    4. Heads: Separate Actor (Policy) and Critic (Value) MLPs.
    """
    def __init__(self, num_node_features, num_actions, hidden_dim=64, edge_dim=4):
        super(MoleculeAgent, self).__init__()
        
        # 1. Feature Extraction (GNN Backbone)
        self.gin_mlp = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(self.gin_mlp, edge_dim=edge_dim) 
        
        # Multi-Head Attention for chemical nuance
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=edge_dim)
        # Input to conv3 is hidden_dim * 4 (due to concat), output is compressed to hidden_dim * 2
        self.conv3 = GATv2Conv(hidden_dim * 4, hidden_dim * 2, heads=2, concat=False, edge_dim=edge_dim)
        
        # 2. Global Pooling (Graph -> Vector)
        # Attentional Aggregation allows the network to "ignore" boring parts of the molecule
        self.pool_gate = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        self.pool = AttentionalAggregation(self.pool_gate)

        # 3. Decision Heads
        self.actor = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, edge_index, edge_attr, batch, action_mask=None):
        """
        Forward pass for PPO.
        Args:
            x, edge_index, edge_attr: PyG graph tensors.
            batch: Tensor indicating which node belongs to which graph in the batch.
            action_mask: Boolean mask for invalid actions.
        """
        # GNN Layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        
        # Pooling
        emb = self.pool(x, batch) 

        # Heads
        logits = self.actor(emb)
        val = self.critic(emb)

        # Masking Invalid Actions
        if action_mask is not None:
            invalid_mask = ~action_mask
            min_val = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(invalid_mask, min_val)
            
        return logits, val