import torch
import numpy as np
import pandas as pd
import time
import os
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import dijkstra

# وارد کردن مدل و تنظیمات از فایل اصلی
from spanner_pipeline import GeometricEdgeSAGE, glv_repair_directed, CITIES

def run_aggressive_benchmark():
    print("--- 🔬 Running Aggressive Sparsification Benchmark (Priority 1) ---")
    weights_path = "best_base_model.pt"
    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    
    # لود شبکه جاده‌ای آیندهوون
    G = ox.graph_from_place(CITIES["Eindhoven"], network_type="drive")
    edge_list = list(G.edges(data=True))
    node_map = {n: i for i, n in enumerate(G.nodes())}
    
    # آماده‌سازی ویژگی‌ها برای GNN
    raw_node = np.array([[G.in_degree(n), G.out_degree(n), 0.1] for n in G.nodes()])
    x = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx = torch.tensor([[node_map[u] for u, v, d in edge_list], [node_map[v] for u, v, d in edge_list]], dtype=torch.long)
    raw_ea = torch.tensor(StandardScaler().fit_transform(np.array([[d['length']] for u, v, d in edge_list])), dtype=torch.float)
    
    # استنتاج概率 GNN
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x, edge_idx, raw_ea)).numpy().flatten()
        
    # تست پله‌ای آستانه‌ها برای رسیدن به هرس‌های سنگین
    thresholds = [0.40, 0.80, 0.90, 0.95, 0.98]
    
    results = []
    for tau in thresholds:
        # ۱. هرس اولیه توسط GNN بر اساس آستانه جدید
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
        
        # ۲. اجرای ترمیم هوشمند مستقل مکانی روی گراف به شدت هرس شده
        t0 = time.time()
        G_repaired, repairs = glv_repair_directed(G_sparse, removed_edges_info, t_limit=1.5)
        repair_time = time.time() - t0
        
        # محاسبه درصد هرس نهایی باقی‌مانده (حفظ شده)
        final_sparsification = ((len(removed_edges_info) - len(repairs)) / len(edge_list)) * 100
        
        # محاسبه ضریب کشش نهایی روی گراف هرس شده
        nodes = list(G.nodes())
        sources = np.random.choice(len(nodes), 200, replace=False)
        d_orig = dijkstra(nx.adjacency_matrix(G, weight='length'), directed=True, indices=sources)
        d_span = dijkstra(nx.adjacency_matrix(G_repaired, weight='length'), directed=True, indices=sources)
        
        stretches = d_span[d_orig > 0] / d_orig[d_orig > 0]
        max_stretch = np.max(stretches[stretches < np.inf]) if np.any(stretches < np.inf) else 1.0
        
        results.append({
            "Threshold (tau)": tau,
            "Initial Pruned (%)": f"{initial_prune_rate:.2f}%",
            "Final Sparsification (%)": f"{final_sparsification:.2f}%",
            "Max Stretch": f"{max_stretch:.4f}",
            "Repair Time (s)": f"{repair_time:.4f}s"
        })
        
    df_res = pd.DataFrame(results)
    print("\n" + "="*85)
    print("                DEMONSTRATING AGGRESSIVE SPARSIFICATION CAPABILITY")
    print("="*85)
    print(df_res.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    run_aggressive_benchmark()
