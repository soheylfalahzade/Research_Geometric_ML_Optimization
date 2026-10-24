import torch
import numpy as np
import pandas as pd
import time
import os
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import dijkstra

# Import the model and settings from the main file
from spanner_pipeline import GeometricEdgeSAGE, glv_repair_directed, CITIES, safe_weighted_adjacency

def run_high_t_benchmark():
    print("--- 🔬 Running Aggressive Sparsification with Dynamic T-Limits (Priority 1) ---")
    weights_path = "best_base_model.pt"
    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    
    # Load the Eindhoven road network
    G = ox.graph_from_place(CITIES["Eindhoven"], network_type="drive")
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    
    # Prepare the features for GNN
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), 0.1] for n in G.nodes()])
    x = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor([[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]], dtype=torch.long)
    raw_ea = torch.tensor(StandardScaler().fit_transform(np.array([[d['length']] for u, v, d in edge_list])), dtype=torch.float)
    
    # GNN probability inference
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x, edge_idx, raw_ea)).numpy().flatten()
        
    # Sweep the threshold and the stretch factor together
    scenarios = [
        {"tau": 0.80, "t_limit": 2.0},
        {"tau": 0.90, "t_limit": 2.0},
        {"tau": 0.95, "t_limit": 2.5},
        {"tau": 0.98, "t_limit": 2.5},
        {"tau": 0.98, "t_limit": 3.0}
    ]
    
    results = []
    for sc in scenarios:
        tau = sc["tau"]
        t_limit = sc["t_limit"]
        
        # 1. Initial pruning by the GNN
        removed_edges_info = []
        G_sparse = nx.DiGraph()
        G_sparse.add_nodes_from(G.nodes())
        
        for i, (u, v, d) in enumerate(edge_list):
            edge_info = {"u": u, "v": v, "length": d['length'], "idx": i}
            if probs[i] > tau:
                G_sparse.add_edge(u, v, length=d['length'])
            else:
                removed_edges_info.append(edge_info)
                
        initial_prune_rate = (len(removed_edges_info) / len(edge_list)) * 100
        
        # 2. Run spatially independent repair on the heavily pruned graph with the new t
        t0 = time.time()
        G_repaired, repairs = glv_repair_directed(G_sparse, removed_edges_info, t_limit=t_limit)
        repair_time = time.time() - t0
        
        # Compute the final retained percentage after pruning (kept)
        final_sparsification = ((len(removed_edges_info) - len(repairs)) / len(edge_list)) * 100
        
        # Compute the final stretch on random samples
        nodes = list(G.nodes())
        sources = np.random.choice(len(nodes), 200, replace=False)
        d_orig = dijkstra(safe_weighted_adjacency(G, nodelist=list(G.nodes()), weight='length'), directed=True, indices=sources)
        d_span = dijkstra(nx.adjacency_matrix(G_repaired, weight='length'), directed=True, indices=sources)
        
        stretches = d_span[d_orig > 0] / d_orig[d_orig > 0]
        max_stretch = np.max(stretches[stretches < np.inf]) if np.any(stretches < np.inf) else 1.0
        
        results.append({
            "Threshold (tau)": tau,
            "Stretch Limit (t)": t_limit,
            "Initial Pruned (%)": f"{initial_prune_rate:.2f}%",
            "Final Sparsification (%)": f"{final_sparsification:.2f}%",
            "Max Stretch": f"{max_stretch:.4f}",
            "Repair Time (s)": f"{repair_time:.4f}s"
        })
        
    df_res = pd.DataFrame(results)
    print("\n" + "="*95)
    print("             REAL AGGRESSIVE SPARSIFICATION RESULTS UNDER RELAXED T-LIMITS")
    print("="*95)
    print(df_res.to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    run_high_t_benchmark()
