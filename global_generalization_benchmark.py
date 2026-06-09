import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Prepare Eindhoven Model (The Base)
print("Loading Eindhoven dataset to train the base model...")
df_e = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
model = RandomForestClassifier(n_estimators=127, max_depth=7, min_samples_split=3, class_weight='balanced', random_state=42)
model.fit(df_e[features], df_e["is_spanner_edge"])

# 2. Cities to Benchmark
target_cities = {
    "Eindhoven (Baseline)": "Eindhoven, Netherlands",
    "Manhattan (Grid)": "Manhattan, New York, USA",
    "Paris (Radial)": "Paris, France",
    "Rome (Organic)": "Rome, Italy"
}

results = []

for city_label, city_name in target_cities.items():
    print(f"\n--- Processing: {city_label} ---")
    try:
        # Download & Process
        G = ox.graph_from_place(city_name, network_type="drive")
        G = ox.project_graph(G).to_undirected()
        nodes, edges = ox.graph_to_gdfs(G)
        pagerank = nx.pagerank(G, weight='length')
        
        city_data = []
        for u, v, k, data in G.edges(keys=True, data=True):
            length = data.get("length", 0.0)
            u_x, u_y = nodes.loc[u]['x'], nodes.loc[u]['y']
            v_x, v_y = nodes.loc[v]['x'], nodes.loc[v]['y']
            city_data.append({
                "length": length, "u_degree": G.degree(u), "v_degree": G.degree(v),
                "dx": v_x - u_x, "dy": v_y - u_y,
                "edge_centrality": G.degree(u) * G.degree(v) / (length + 1),
                "u_pagerank": pagerank.get(u, 0.0), "v_pagerank": pagerank.get(v, 0.0)
            })
        
        X_city = pd.DataFrame(city_data)
        preds = model.predict(X_city)
        ratio = (1 - (sum(preds) / len(preds))) * 100
        results.append((city_label, ratio))
        print(f"✔ Sparsification Ratio for {city_label}: {ratio:.2f}%")
    except Exception as e:
        print(f"✖ Could not process {city_label}: {e}")

# 3. Create Comparison Chart
plt.figure(figsize=(10, 6))
labels, values = zip(*results)
bars = plt.bar(labels, values, color=['#3498db', '#e74c3c', '#2ecc71'])
plt.axhline(y=29.88, color='black', linestyle='--', label='Baseline (Eindhoven)') # خط مبنای آیندهوون

plt.title("AI Generalization Across Global Urban Morphologies", fontsize=14)
plt.ylabel("Pruning Efficiency (%)", fontsize=12)
plt.legend()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("global_generalization_benchmark.png", dpi=300)
print("\n✔ Benchmark complete! Chart saved as 'global_generalization_benchmark.png'")