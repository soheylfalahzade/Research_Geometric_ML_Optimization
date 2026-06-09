import pandas as pd
import networkx as nx
import osmnx as ox
import numpy as np

print("--- [PHASE 1] Topological Cleaning & Connectivity ---")
# 1. Load the result we got from the AI
df = pd.read_csv("final_scientific_spanner.csv")

# 2. Re-construct the Graph in NetworkX
G_pruned = nx.Graph()
for _, row in df.iterrows():
    G_pruned.add_edge(int(row['node_u']), int(row['node_v']), length=row['length'])

# 3. Step 1: Remove Dangling Nodes (Degree 1) - Iteratively
initial_edges = G_pruned.number_of_edges()
while True:
    dangling_nodes = [node for node, degree in G_pruned.degree() if degree <= 1]
    if not dangling_nodes:
        break
    G_pruned.remove_nodes_from(dangling_nodes)

print(f"✔ Dangling nodes removed. Edges reduced from {initial_edges} to {G_pruned.number_of_edges()}.")

# 4. Step 2: Ensure Strong Connectivity (Largest Component)
if not nx.is_connected(G_pruned):
    print("⚠ Graph was disconnected! Extracting the Largest Connected Component...")
    largest_cc = max(nx.connected_components(G_pruned), key=len)
    G_pruned = G_pruned.subgraph(largest_cc).copy()

print(f"✔ Final Connected Graph: {G_pruned.number_of_nodes()} nodes, {G_pruned.number_of_edges()} edges.")

# 5. Step 3: Global Connectivity Verification
# تست تصادفی بین 500 زوج گره برای اطمینان از اینکه مسیر در کل شهر وجود دارد
nodes = list(G_pruned.nodes())
reachable = 0
for _ in range(500):
    u, v = np.random.choice(nodes, 2, replace=False)
    if nx.has_path(G_pruned, u, v):
        reachable += 1

print(f"✔ Connectivity Reliability: {(reachable/500)*100:.2f}%")

# 6. Export the CLEAN Scientific result
edges_data = []
for u, v, data in G_pruned.edges(data=True):
    edges_data.append({"node_u": u, "node_v": v, "length": data['length']})

df_clean = pd.DataFrame(edges_data)
df_clean.to_csv("Q1_CLEAN_Spanner.csv", index=False)
print("✔ Q1_CLEAN_Spanner.csv saved. This is your Gold Standard dataset.")