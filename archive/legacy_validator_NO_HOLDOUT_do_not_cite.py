import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
df = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
X = df[features]
y = df["is_spanner_edge"]

# 2. Train Optimized Model (Using Genetic Algorithm result)
print("Training final model with GA-optimized parameters...")
best_model = RandomForestClassifier(
    n_estimators=127, 
    max_depth=7, 
    min_samples_split=3, 
    random_state=42, 
    class_weight='balanced'
)
best_model.fit(X, y)

# 3. Predict Pruning for all edges
df['predicted_spanner'] = best_model.predict(X)

# 4. Analyze Results
total_edges = len(df)
pruned_edges = total_edges - df['predicted_spanner'].sum()
pruning_ratio = (pruned_edges / total_edges) * 100

print("\n================== FINAL OPTIMIZATION REPORT ==================")
print(f"Total Edges Processed: {total_edges}")
print(f"Edges Pruned by Model: {pruned_edges}")
print(f"Pruning Efficiency: {pruning_ratio:.2f}%")
print("===============================================================")

# 5. Export for visualization
df_spanner = df[df['predicted_spanner'] == 1]
df_spanner.to_csv("optimized_spanner_edges.csv", index=False)
print("Optimized spanner edges saved to optimized_spanner_edges.csv.")