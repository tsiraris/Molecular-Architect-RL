import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class MoleculeAgent(nn.Module):
    def __init__(self, num_node_features, num_actions, hidden_dim=64):
        super(MoleculeAgent, self).__init__()
        
        # 1. GRAPH ENCODER - GCNConv layers allow information to flow between atoms (nodes)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 2)

        # 2. ACTOR HEAD - Decides what to do next based on the graph embedding
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)  # Output: Logits for each possible action
        )

        # 3. CRITIC HEAD - Predicts how good the current state is (scalar value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: Single score (Value)
        )

    def forward(self, x, edge_index, batch, action_mask=None):
        """
        x: Node features (Atom types)
        edge_index: Graph connectivity (Bonds)
        batch: Keeps track of which nodes belong to which molecule in a batch
        """
        
        # --- Stage 1: Message Passing (Graph Convolution) ---
        # Layer 1 - One hop knowledge aggregation
        x = self.conv1(x, edge_index)
        x = F.relu(x)   # --> x is a context-aware representation of each atom now
        
        # Layer 2 - Two hops
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)  # [Num_Atoms, Hidden_Dim*2]

        # --- Stage 2: Global Pooling (Readout) ---
        # Collapse the whole graph (molecule) into a single vector - all atomic information into a "Molecule Vector"
        graph_embedding = global_mean_pool(x, batch) # [Num_Molecules, Hidden_Dim*2]

        # --- Stage 3: Heads ---
        action_logits = self.actor(graph_embedding) # [Num_Molecules, Num_Actions] - Will be used for action selection
        state_value = self.critic(graph_embedding)  # [Num_Molecules, 1] - Value estimation/Baseline expectation

        # --- Stage 4: Action Masking ---
        if action_mask is not None:
            # Ensure mask is broadcastable if you are using batches later
            # Invert mask: We want to find the INVALID indices (1=Valid, 0=Invalid)
            action_logits = action_logits.masked_fill(~action_mask, -1e9) # Ensures softmax gives ~0 prob to invalid actions
            
        return action_logits, state_value 

# --- VERIFICATION BLOCK --- Debug without training
if __name__ == "__main__":
    print("🧠 INITIALIZING GNN AGENT...")
    
    # 1. Setup Dummy Data
    # For a dummy molecule with 3 atoms (Nodes) and 2 bonds (Edges)
    dummy_x = torch.tensor([
        [1.0, 0.0, 0.0], # Atom 1 (Carbon)
        [0.0, 1.0, 0.0], # Atom 2 (Nitrogen)
        [1.0, 0.0, 0.0]  # Atom 3 (Carbon)
    ], dtype=torch.float)
    
    # Edges (Connections: 0-1 and 1-2) using Coordinate Format (COO)
    dummy_edge_index = torch.tensor([
        [0, 1, 1, 2], # Source Nodes
        [1, 0, 2, 1]  # Target Nodes (Undirected = both ways)
    ], dtype=torch.long)
    
    # Batch Vector (Assuming all nodes belong to Molecule 0)
    dummy_batch = torch.tensor([0, 0, 0], dtype=torch.long)

    # 2. Check Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # 3. Initialize Model
    # Input features = 3 (size of one-hot vector), Actions = 5 (e.g., Add C, Add N, etc.)
    agent = MoleculeAgent(num_node_features=3, num_actions=5).to(device)
    
    # Move data to GPU
    dummy_x = dummy_x.to(device)
    dummy_edge_index = dummy_edge_index.to(device)
    dummy_batch = dummy_batch.to(device)

    # 4. Forward Pass
    logits, value = agent(dummy_x, dummy_edge_index, dummy_batch)
    
    print("\n✅ FORWARD PASS SUCCESSFUL")
    print(f"   Action Logits Shape: {logits.shape} (Should be [1, {agent.num_actions}])")
    print(f"   Critic Value: {value.item():.4f}")
    
    print("\n   The GNN successfully processed the molecular graph structure!")