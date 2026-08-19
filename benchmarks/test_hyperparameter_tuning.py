import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# وارد کردن توابع پایه‌ای از فایل اصلی
from spanner_pipeline import GeometricEdgeSAGE, GLV_T_LIMIT, NUM_DIJKSTRA_THREADS, _check_one_directed_edge, compute_global_stretch_scipy

NEW_FINETUNE_EPOCHS = 5       # ۵ اپوک فین‌تیون نرم
NEW_FINETUNE_LR     = 0.0001  # نرخ یادگیری ملایم

def glv_repair_batched(G_sparse, removed_edges_info, t_limit=GLV_T_LIMIT, batch_size=250):
    repaired_G = G_sparse.copy()
    repairs    = []
    candidates = sorted(removed_edges_info, key=lambda e: e["length"])
    num_candidates = len(candidates)
    
    for start_idx in range(0, num_candidates, batch_size):
        batch = candidates[start_idx : start_idx + batch_size]
        args_list = [(repaired_G, edge, t_limit) for edge in batch]
        batch_needs_repair = {}
        
        with ThreadPoolExecutor(max_workers=NUM_DIJKSTRA_THREADS) as pool:
            for idx, needs in pool.map(_check_one_directed_edge, args_list):
                batch_needs_repair[idx] = needs
                
        for edge in batch:
            if batch_needs_repair.get(edge["idx"], False):
                repaired_G.add_edge(edge["u"], edge["v"], length=edge["length"])
                repairs.append(edge["idx"])
                
    return repaired_G, repairs

def test_threshold_grid_search():
    print("--- Initiating Systematic Threshold Grid Search on [Eindhoven] ---")
    weights_path = 'best_base_model.pt'
    
    # دانلود گراف و پل‌های هندسی فقط یک‌بار برای افزایش شدید سرعت اجرای تست
    G = ox.graph_from_place("Eindhoven, Netherlands", network_type="drive")
    G = ox.project_graph(G).to_undirected()
    
    bridges = {tuple(sorted((u, v))) for u, v in nx.bridges(G)}
    pagerank  = nx.pagerank(G, weight="length")
    edge_list = list(G.edges(keys=True, data=True))
    node_map  = {n: i for i, n in enumerate(G.nodes())}
    raw_node  = np.array([[G.degree(n), pagerank.get(n, 1e-4)] for n in G.nodes()])
    x_local   = torch.tensor(StandardScaler().fit_transform(raw_node), dtype=torch.float)
    edge_idx  = torch.tensor(
        [[node_map[u] for u, v, k, d in edge_list],
         [node_map[v] for u, v, k, d in edge_list]], dtype=torch.long
    )
    raw_ea        = np.array([[d["length"]] for u, v, k, d in edge_list])
    edge_attr_loc = torch.tensor(StandardScaler().fit_transform(raw_ea), dtype=torch.float)
    
    # تعریف آستانه‌های هدف برای اسکن سیستماتیک
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    grid_results = []
    
    for theta in thresholds:
        print(f"\nEvaluating Threshold (theta) = {theta:.2f}...")
        
        # لود کردن مجدد وزن‌های پایه برای استقلال هر آزمایش از آزمایش قبلی
        model = GeometricEdgeSAGE()
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
        model.eval()
        
        # ۱. استنتاج دور اول با آستانه فعلی
        with torch.no_grad():
            probs = torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten()
            
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(G.nodes())
        removed_edges = []
        
        for i, (u, v, k, d) in enumerate(edge_list):
            edge_key = tuple(sorted((u, v)))
            is_bridge = edge_key in bridges
            is_low_degree = (G.degree(u) <= 1) or (G.degree(v) <= 1)
            
            if probs[i] > theta or is_bridge or is_low_degree:
                G_sparse.add_edge(u, v, length=d["length"])
            else:
                removed_edges.append({
                    "u": u, "v": v, "length": d["length"], "prob": probs[i], "idx": i
                })
                
        G_repaired, repaired_indices = glv_repair_batched(G_sparse, removed_edges)
        
        # ۲. فین‌تیون نرم
        optimizer = torch.optim.Adam(model.parameters(), lr=NEW_FINETUNE_LR)
        repaired_set = set(repaired_indices)
        y_vals = []
        for i in range(len(edge_list)):
            if i in repaired_set:
                y_vals.append(1.0)
            elif probs[i] > theta:
                y_vals.append(1.0)
            else:
                y_vals.append(0.0)
        y_feedback = torch.tensor(y_vals, dtype=torch.float).view(-1, 1)
        
        model.train()
        for ep in range(NEW_FINETUNE_EPOCHS):
            optimizer.zero_grad()
            out = model(x_local, edge_idx, edge_attr_loc)
            loss = F.binary_cross_entropy_with_logits(out, y_feedback)
            loss.backward()
            optimizer.step()
            
        # ۳. استنتاج نهایی با آستانه کالیبره‌شده
        model.eval()
        with torch.no_grad():
            final_probs = torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten()
            
        G_opt = nx.Graph()
        G_opt.add_nodes_from(G.nodes())
        removed_final = []
        
        for i, (u, v, k, d) in enumerate(edge_list):
            edge_key = tuple(sorted((u, v)))
            is_bridge = edge_key in bridges
            is_low_degree = (G.degree(u) <= 1) or (G.degree(v) <= 1)
            
            if final_probs[i] > theta or is_bridge or is_low_degree:
                G_opt.add_edge(u, v, length=d["length"])
            else:
                removed_final.append({
                    "u": u, "v": v, "length": d["length"], "prob": final_probs[i], "idx": i
                })
                
        G_final, final_repairs = glv_repair_batched(G_opt, removed_final)
        
        # ارزیابی ۱۰۰,۰۰۰ دایجسترا به صورت فوق‌سریع در SciPy
        stretches = compute_global_stretch_scipy(G, G_final, num_samples=100000)
        
        sparsification = (1 - G_final.number_of_edges() / G.number_of_edges()) * 100
        mean_s = np.mean(stretches)
        max_s = np.max(stretches)
        
        grid_results.append({
            'Threshold (theta)': f"{theta:.2f}",
            'Sparsification': f"{sparsification:.2f}%",
            'Avg Stretch': f"{mean_s:.4f}",
            'Max Stretch': f"{max_s:.4f}",
            'Repairs': len(final_repairs)
        })
        
    print("\n" + "="*75)
    print("                 SYSTEMATIC SENSITIVITY GRID SEARCH RESULTS")
    print("="*75)
    print(pd.DataFrame(grid_results).to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    test_threshold_grid_search()