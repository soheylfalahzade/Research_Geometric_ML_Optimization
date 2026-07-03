import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load Data & Scale Invariant Engineering
print("Engineering Scale-Invariant Geometric Features...")
df = pd.read_csv("spanner_dataset_pro.csv")

# Standardizing features for Scale Invariance (Important for Q1)
scaler = StandardScaler()
features_list = ["length", "u_degree", "v_degree", "edge_centrality", "u_pagerank", "v_pagerank"]
df[features_list] = scaler.fit_transform(df[features_list])

nodes_unique = pd.concat([df['node_u'], df['node_v']]).unique()
node_map = {node: i for i, node in enumerate(nodes_unique)}
edge_index = torch.tensor([[node_map[u] for u in df['node_u']], [node_map[v] for v in df['node_v']]], dtype=torch.long)

# Mapping node features
node_feat_map = {}
for _, row in df.iterrows():
    node_feat_map[node_map[row['node_u']]] = [row['u_degree'], row['u_pagerank']]
    node_feat_map[node_map[row['node_v']]] = [row['v_degree'], row['v_pagerank']]

x = torch.tensor([node_feat_map.get(i, [0.0, 0.0]) for i in range(len(nodes_unique))], dtype=torch.float)
y = torch.tensor(df['is_spanner_edge'].values, dtype=torch.float).view(-1, 1)
edge_attr = torch.tensor(df['length'].values, dtype=torch.float).view(-1, 1)
importance = torch.tensor(df['edge_centrality'].values, dtype=torch.float).view(-1, 1)

# 2. Inductive Architecture
class GeometricEdgeSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GeometricEdgeSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        u, v = edge_index
        edge_repr = torch.cat([h[u], h[v], edge_attr], dim=-1)
        return self.classifier(edge_repr)

# 3. Balanced Q1 Loss: Penalty + Sparsity Pressure
def q1_balanced_loss(pred_logits, target, importance, lambda_p=2.0, alpha_s=0.5):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    probs = torch.sigmoid(pred_logits)
    
    # Penalty: Don't cut bridges (False Negatives)
    penalty = lambda_p * (importance * target * (1 - probs)).mean()
    
    # Sparsity Pressure: Encourage pruning (Don't keep everything)
    sparsity_loss = alpha_s * probs.mean()
    
    return bce + penalty + sparsity_loss

model = GeometricEdgeSAGE(in_channels=2, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print(f"Starting Training: Finding the Pareto Frontier...")
for epoch in range(301):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, edge_attr)
    loss = q1_balanced_loss(out, y, importance)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0: print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# 4. Final Result
model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(x, edge_index, edge_attr)).numpy() > 0.5).astype(int)
    df['final_pred'] = preds

ratio = (1 - (df['final_pred'].sum() / len(df))) * 100
print(f"\n================== Q1 DIAMOND RESULTS ==================")
print(f"Final Pruning Efficiency: {ratio:.2f}%")
print(f"Geometric Bias: Scale-Invariant & Sparsity-Aware")
print(f"Status: Target 15-25% Pruning Reached?")
print("==========================================================")