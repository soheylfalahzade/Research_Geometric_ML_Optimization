import pandas as pd
import networkx as nx
import osmnx as ox
import time
import numpy as np

# 1. Load the Cleaned Data (from Eindhoven as example)
print("Loading Eindhoven clean graph for routing benchmark...")
df_clean = pd.read_csv("Q1_CLEAN_Spanner.csv")
G_orig = ox.graph_from_place("Eindhoven, Netherlands", network_type="drive")
G_orig = ox.project_graph(G_orig).to_undirected()

# 2. Build the Optimized Graph
G_opt = nx.Graph()
G_opt.add_nodes_from(G_orig.nodes())
for _, row in df_clean.iterrows():
    G_opt.add_edge(int(row['node_u']), int(row['node_v']), length=row['length'])

# Ensure we only test nodes present in both
common_nodes = list(set(G_orig.nodes()) & set(G_opt.nodes()))

print(f"Starting Speed Test: 100 Random Queries...")
t_orig_total = 0
t_opt_total = 0
stretch_factors = []

for i in range(100):
    u, v = np.random.choice(common_nodes, 2, replace=False)
    
    # Original Graph Search
    start = time.time()
    try:
        d_orig = nx.shortest_path_length(G_orig, u, v, weight='length')
        t_orig_total += (time.time() - start)
        
        # Optimized Graph Search
        start = time.time()
        d_opt = nx.shortest_path_length(G_opt, u, v, weight='length')
        t_opt_total += (time.time() - start)
        
        stretch_factors.append(d_opt / d_orig)
    except:
        continue

print("\n================== FINAL ROUTING BENCHMARK ==================")
print(f"Avg Latency (Original): {(t_orig_total/100)*1000:.4f} ms")
print(f"Avg Latency (Optimized): {(t_opt_total/100)*1000:.4f} ms")
print(f"SEARCH SPEEDUP: {(t_orig_total/t_opt_total):.2f}x")
print(f"ACTUAL AVG STRETCH (t): {np.mean(stretch_factors):.4f}")
print(f"THEORETICAL LIMIT (t_max): 2.0000")
print("=============================================================")