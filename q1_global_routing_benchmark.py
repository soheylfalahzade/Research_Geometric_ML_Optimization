import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# 1. Train the Universal Model on Eindhoven
print("Training Universal Pruning Model...")
df_e = pd.read_csv("spanner_dataset_pro.csv")
features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
model = RandomForestClassifier(n_estimators=127, max_depth=7, min_samples_split=3, class_weight='balanced', random_state=42)
model.fit(df_e[features], df_e["is_spanner_edge"])

target_cities = {
    "Eindhoven": "Eindhoven, Netherlands",
    "Manhattan": "Manhattan, New York, USA",
    "Paris": "Paris, France",
    "Rome": "Rome, Italy"
}

global_results = []

for city_name, city_query in target_cities.items():
    print(f"\n--- Benchmarking: {city_name} ---")
    
    # Download & Process
    G_orig = ox.graph_from_place(city_query, network_type="drive")
    G_orig = ox.project_graph(G_orig).to_undirected()
    nodes, edges = ox.graph_to_gdfs(G_orig)
    pagerank = nx.pagerank(G_orig, weight='length')
    
    # AI Pruning
    city_data = []
    for u, v, k, data in G_orig.edges(keys=True, data=True):
        length = data.get("length", 0.0)
        edge_centrality = G_orig.degree(u) * G_orig.degree(v) / (length + 1)
        city_data.append({
            "length": length, "u_degree": G_orig.degree(u), "v_degree": G_orig.degree(v),
            "dx": nodes.loc[v]['x'] - nodes.loc[u]['x'], "dy": nodes.loc[v]['y'] - nodes.loc[u]['y'],
            "edge_centrality": edge_centrality,
            "u_pagerank": pagerank.get(u, 0.0), "v_pagerank": pagerank.get(v, 0.0)
        })
    
    preds = model.predict(pd.DataFrame(city_data))
    
    # Build Optimized Graph & Clean
    G_opt = nx.Graph()
    G_opt.add_nodes_from(G_orig.nodes())
    edge_list = list(G_orig.edges(data=True))
    for i, (u, v, data) in enumerate(edge_list):
        if preds[i] == 1:
            G_opt.add_edge(u, v, length=data['length'])
    
    # Topological Cleaning (Remove Dangling & Disconnected)
    while True:
        dangling = [n for n, d in G_opt.degree() if d <= 1]
        if not dangling: break
        G_opt.remove_nodes_from(dangling)
    
    if not nx.is_connected(G_opt):
        largest_cc = max(nx.connected_components(G_opt), key=len)
        G_opt = G_opt.subgraph(largest_cc).copy()

    # Speed Test (1000 Samples)
    common_nodes = list(set(G_orig.nodes()) & set(G_opt.nodes()))
    t_orig, t_opt = 0, 0
    stretches = []
    
    print(f"Testing 1000 random routes in {city_name}...")
    for _ in range(1000):
        u, v = np.random.choice(common_nodes, 2, replace=False)
        try:
            start = time.time()
            d_o = nx.shortest_path_length(G_orig, u, v, weight='length')
            t_orig += (time.time() - start)
            
            start = time.time()
            d_p = nx.shortest_path_length(G_opt, u, v, weight='length')
            t_opt += (time.time() - start)
            stretches.append(d_p / d_o)
        except: continue

    speedup = t_orig / t_opt if t_opt > 0 else 0
    avg_stretch = np.mean(stretches)
    
    global_results.append({
        "City": city_name,
        "Original Edges": G_orig.number_of_edges(),
        "Pruned Edges": G_opt.number_of_edges(),
        "Speedup": f"{speedup:.1f}x",
        "Avg Stretch": f"{avg_stretch:.3f}"
    })

# 3. Final Summary Table
print("\n" + "="*60)
print("FINAL Q1 GLOBAL BENCHMARK SUMMARY")
print("="*60)
summary_df = pd.DataFrame(global_results)
print(summary_df.to_string(index=False))
print("="*60)