import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np

# 1. Load Data
print("Loading real topological features for GNN...")
df = pd.read_csv("spanner_dataset_pro.csv")

nodes_unique = pd.concat([df['node_u'], df['node_v']]).unique()
node_map = {node: i for i, node in enumerate(nodes_unique)}
edge_index = torch.tensor([[node_map[u] for u in df['node_u']], [node_map[v] for v in df['node_v']]], dtype=torch.long)

# Prepare Node Features
node_feat_map = {}
for _, row in df.iterrows():
    node_feat_map[node_map[row['node_u']]] = [row['u_degree'], row['u_pagerank']]
    node_feat_map[node_map[row['node_v']]] = [row['v_degree'], row['v_pagerank']]
x = torch.tensor([node_feat_map.get(i, [2.0, 0.0001]) for i in range(len(nodes_unique))], dtype=torch.float)

# Targets & Attributes
y = torch.tensor(df['is_spanner_edge'].values, dtype=torch.float).view(-1, 1)
edge_attr = torch.tensor(df['length'].values, dtype=torch.float).view(-1, 1)

# Handle Class Imbalance
num_pos = y.sum()
num_neg = len(y) - num_pos
pos_weight = torch.tensor([num_neg / num_pos])

# 2. Advanced EdgeSAGE Architecture
class EdgeSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgeSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) 
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        u, v = edge_index
        edge_repr = torch.cat([h[u], h[v], edge_attr], dim=-1)
        return self.classifier(edge_repr)

# 3. Training Setup
device = torch.device('cpu')
model = EdgeSAGE(in_channels=2, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"Starting GNN Training with Early Stopping (Max 500 Epochs)...")
best_loss = float('inf')
patience = 40 
counter = 0

for epoch in range(501):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_attr)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    # Corrected state_dict() call
    if loss < best_loss:
        best_loss = loss
        counter = 0
        torch.save(model.state_dict(), "best_gnn_model.pt")
    else:
        counter += 1
        
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")
    
    if counter >= patience:
        print(f"--- Early stopping triggered at epoch {epoch} ---")
        break

# 4. Final Inference using the BEST model
model.load_state_dict(torch.load("best_gnn_model.pt"))
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(x, edge_index, edge_attr))
    df['gnn_predicted'] = (predictions.numpy() > 0.5).astype(int)

gnn_ratio = (1 - (df['gnn_predicted'].sum() / len(df))) * 100
print(f"\n================== FINAL OPTIMIZED GNN ==================")
print(f"GNN Pruning Efficiency: {gnn_ratio:.2f}%")
print(f"Best Training Loss: {best_loss:.4f}")
print(f"Status: Model Architecture ready for Q1 Manuscript.")
print("==========================================================")