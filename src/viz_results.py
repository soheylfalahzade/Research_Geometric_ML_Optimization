import pandas as pd
import osmnx as ox
import folium
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
df = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
X = df[features]
y = df["is_spanner_edge"]

# 2. Re-train/Predict to ensure column exists
model = RandomForestClassifier(n_estimators=127, max_depth=7, min_samples_split=3, random_state=42, class_weight='balanced')
model.fit(X, y)
df['predicted_spanner'] = model.predict(X)

spanner_edges = df[df['predicted_spanner'] == 1]

# 3. Load base graph
print("Loading map data...")
G = ox.graph_from_place("Eindhoven, Netherlands", network_type="drive")

# 4. Create Folium Map
m = folium.Map(location=[51.4416, 5.4697], zoom_start=13, tiles="cartodbpositron")

# 5. Plot Spanner Edges
print(f"Plotting {len(spanner_edges)} optimized edges...")
for _, row in spanner_edges.iterrows():
    u, v = int(row['node_u']), int(row['node_v'])
    if G.has_edge(u, v):
        line = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        folium.PolyLine(line, color="blue", weight=2.5, opacity=0.8).add_to(m)

m.save("final_pruned_graph.html")
print("Interactive Pruned Graph saved to final_pruned_graph.html")