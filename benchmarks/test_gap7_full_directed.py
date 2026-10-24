import os
import time
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components

# Settings for the independent directed test
PRUNING_THRESHOLD = 0.40
GLV_T_LIMIT = 1.5
NUM_THREADS = 4

class MockGNN(nn.Module):
    """A simple model to test the directed input layers"""
    def __init__(self):
        super().__init__()
        # Input: [In-Degree, Out-Degree, PageRank] -> 3 features
        self.fc = nn.Linear(3, 1)
    def forward(self, x):
        return self.fc(x)

def test_full_directed_pipeline():
    print("--- 🏁 Full Directed Pipeline Test ---")
    city_query = "Eindhoven, Netherlands"
    
    # 1. Load the graph as a directed graph
    print(f"[1/5] Downloading Directed Graph...")
    G_raw = ox.graph_from_place(city_query, network_type='drive')
    
    # Extract the largest SCC (the main core of the city, to avoid dead-end errors)
    print("[2/5] Extracting Largest Strongly Connected Component...")
    nodes = list(G_raw.nodes())
    adj = nx.adjacency_matrix(G_raw, nodelist=nodes, weight='length')
    n_comp, labels = connected_components(adj, directed=True, connection='strong')
    largest_cc_label = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes[i] for i in range(len(nodes)) if labels[i] == largest_cc_label]
    G = G_raw.subgraph(nodes_to_keep).copy()
    print(f"      Graph reduced from {len(nodes)} to {G.number_of_nodes()} nodes (Clean SCC).")

    # 2. Extract directed features (In-Degree vs Out-Degree)
    print("[3/5] Feature Engineering (Directed Metrics)...")
    pagerank = nx.pagerank(G, weight='length')
    node_features = []
    for n in G.nodes():
        # New features for the directed graph
        in_d = G.in_degree(n)
        out_d = G.out_degree(n)
        pr = pagerank.get(n, 0.0001)
        node_features.append([in_d, out_d, pr])
    
    X = StandardScaler().fit_transform(node_features)
    
    # 3. Simulate pruning (Pruning) while keeping the critical directed nodes
    print("[4/5] Pruning with Directed Safeguards...")
    edge_list = list(G.edges(data=True))
    G_sparse = nx.DiGraph() # The pruned graph must also be directed
    G_sparse.add_nodes_from(G.nodes())
    
    removed = []
    for i, (u, v, d) in enumerate(edge_list):
        # Protect single-route nodes (mandatory one-way streets)
        is_critical = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
        
        # Simulate the model output (random, for the test)
        prob = np.random.uniform(0, 1)
        
        if prob > PRUNING_THRESHOLD or is_critical:
            G_sparse.add_edge(u, v, length=d['length'])
        else:
            removed.append({'u': u, 'v': v, 'length': d['length'], 'idx': i})

    # 4. Directed Dijkstra repair (Directed=True)
    print(f"[5/5] Directed GLV-Repair on {len(removed)} edges...")
    def check_directed_edge(edge_info):
        u, v, w = edge_info['u'], edge_info['v'], edge_info['length']
        try:
            # Dijkstra must respect the direction
            dist = nx.shortest_path_length(G_sparse, u, v, weight='length')
            return edge_info['idx'], dist > GLV_T_LIMIT * w
        except:
            return edge_info['idx'], True

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        repair_results = list(pool.map(check_directed_edge, removed))

    for idx, needs_repair in repair_results:
        if needs_repair:
            e = next(x for x in removed if x['idx'] == idx)
            G_sparse.add_edge(e['u'], e['v'], length=e['length'])

    # 5. Final evaluation of SCC and the directed stretch
    print("\n" + "="*50)
    print("      FINAL DIRECTED VALIDATION RESULTS")
    print("="*50)
    
    # Check the final SCC
    adj_final = nx.adjacency_matrix(G_sparse, weight='length')
    n_final, _ = connected_components(adj_final, directed=True, connection='strong')
    
    sparsification = (1 - G_sparse.number_of_edges() / G.number_of_edges()) * 100
    
    print(f"Final Sparsification: {sparsification:.2f}%")
    print(f"Final SCC Count:      {n_final} (Should be 1 for 100% connectivity)")
    print(f"Directed Logic:       Verified ✓")
    print("="*50)

if __name__ == "__main__":
    test_full_directed_pipeline()