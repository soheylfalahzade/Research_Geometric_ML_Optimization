import pandas as pd
import osmnx as ox
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("--- [GLOBAL GENERALIZATION TEST: MANHATTAN] ---")

# 1. Load your optimized model (The one trained on Eindhoven)
df_eindhoven = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
X_train = df_eindhoven[features]
y_train = df_eindhoven["is_spanner_edge"]

print("Re-training model on Eindhoven to test transferability...")
model = RandomForestClassifier(n_estimators=127, max_depth=7, min_samples_split=3, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 2. Download Manhattan Graph
print("Downloading Manhattan (New York) grid-based network...")
G_man = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")
G_man = ox.project_graph(G_man).to_undirected()

# 3. Extract Features for Manhattan
print("Extracting features for Manhattan...")
nodes, edges = ox.graph_to_gdfs(G_man)
pagerank = nx.pagerank(G_man, weight='length')

man_data = []
for u, v, k, data in G_man.edges(keys=True, data=True):
    length = data.get("length", 0.0)
    u_x, u_y = nodes.loc[u]['x'], nodes.loc[u]['y']
    v_x, v_y = nodes.loc[v]['x'], nodes.loc[v]['y']
    edge_centrality = G_man.degree(u) * G_man.degree(v) / (length + 1)
    
    man_data.append({
        "length": length, "u_degree": G_man.degree(u), "v_degree": G_man.degree(v),
        "dx": v_x - u_x, "dy": v_y - u_y, "edge_centrality": edge_centrality,
        "u_pagerank": pagerank.get(u, 0.0), "v_pagerank": pagerank.get(v, 0.0)
    })

X_man = pd.DataFrame(man_data)

# 4. Predict Pruning on Manhattan
print("Predicting Manhattan sparsification using Eindhoven-trained model...")
predictions = model.predict(X_man)
sparsification = (1 - (sum(predictions) / len(predictions))) * 100

print("\n================== GENERALIZATION RESULTS ==================")
print(f"Manhattan Original Edges: {len(X_man)}")
print(f"Manhattan Pruned Edges: {len(X_man) - sum(predictions)}")
print(f"Manhattan Sparsification Ratio: {sparsification:.2f}%")
print("============================================================")