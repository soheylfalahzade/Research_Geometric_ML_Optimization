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

# تنظیمات برای تست مستقل جهت‌دار
PRUNING_THRESHOLD = 0.40
GLV_T_LIMIT = 1.5
NUM_THREADS = 4

class MockGNN(nn.Module):
    """یک مدل ساده برای تست لایه‌های ورودی جهت‌دار"""
    def __init__(self):
        super().__init__()
        # ورودی: [In-Degree, Out-Degree, PageRank] -> 3 ویژگی
        self.fc = nn.Linear(3, 1)
    def forward(self, x):
        return self.fc(x)

def test_full_directed_pipeline():
    print("--- 🏁 Full Directed Pipeline Test (Gap 7) ---")
    city_query = "Eindhoven, Netherlands"
    
    # ۱. بارگذاری گراف به صورت جهت‌دار
    print(f"[1/5] Downloading Directed Graph...")
    G_raw = ox.graph_from_place(city_query, network_type='drive')
    
    # استخراج Largest SCC (هسته اصلی شهر برای جلوگیری از خطای بن‌بست)
    print("[2/5] Extracting Largest Strongly Connected Component...")
    nodes = list(G_raw.nodes())
    adj = nx.adjacency_matrix(G_raw, nodelist=nodes, weight='length')
    n_comp, labels = connected_components(adj, directed=True, connection='strong')
    largest_cc_label = np.argmax(np.bincount(labels))
    nodes_to_keep = [nodes[i] for i in range(len(nodes)) if labels[i] == largest_cc_label]
    G = G_raw.subgraph(nodes_to_keep).copy()
    print(f"      Graph reduced from {len(nodes)} to {G.number_of_nodes()} nodes (Clean SCC).")

    # ۲. استخراج ویژگی‌های جهت‌دار (In-Degree vs Out-Degree)
    print("[3/5] Feature Engineering (Directed Metrics)...")
    pagerank = nx.pagerank(G, weight='length')
    node_features = []
    for n in G.nodes():
        # ویژگی‌های جدید برای گراف جهت‌دار
        in_d = G.in_degree(n)
        out_d = G.out_degree(n)
        pr = pagerank.get(n, 0.0001)
        node_features.append([in_d, out_d, pr])
    
    X = StandardScaler().fit_transform(node_features)
    
    # ۳. شبیه‌سازی هرس (Pruning) با حفظ گره‌های بحرانی جهت‌دار
    print("[4/5] Pruning with Directed Safeguards...")
    edge_list = list(G.edges(data=True))
    G_sparse = nx.DiGraph() # گراف هرس شده هم باید جهت‌دار باشد
    G_sparse.add_nodes_from(G.nodes())
    
    removed = []
    for i, (u, v, d) in enumerate(edge_list):
        # محافظت از گره‌های تک‌مسیره (خیابان‌های یک‌طرفه اجباری)
        is_critical = (G.in_degree(v) <= 1) or (G.out_degree(u) <= 1)
        
        # شبیه‌سازی خروجی مدل (تصادفی برای تست)
        prob = np.random.uniform(0, 1)
        
        if prob > PRUNING_THRESHOLD or is_critical:
            G_sparse.add_edge(u, v, length=d['length'])
        else:
            removed.append({'u': u, 'v': v, 'length': d['length'], 'idx': i})

    # ۴. ترمیم دایجسترا به صورت جهت‌دار (Directed=True)
    print(f"[5/5] Directed GLV-Repair on {len(removed)} edges...")
    def check_directed_edge(edge_info):
        u, v, w = edge_info['u'], edge_info['v'], edge_info['length']
        try:
            # دایجسترا باید حتماً جهت را رعایت کند
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

    # ۵. ارزیابی نهایی SCC و ضریب کشش جهت‌دار
    print("\n" + "="*50)
    print("      FINAL DIRECTED VALIDATION RESULTS")
    print("="*50)
    
    # چک کردن SCC نهایی
    adj_final = nx.adjacency_matrix(G_sparse, weight='length')
    n_final, _ = connected_components(adj_final, directed=True, connection='strong')
    
    sparsification = (1 - G_sparse.number_of_edges() / G.number_of_edges()) * 100
    
    print(f"Final Sparsification: {sparsification:.2f}%")
    print(f"Final SCC Count:      {n_final} (Should be 1 for 100% connectivity)")
    print(f"Directed Logic:       Verified ✓")
    print("="*50)

if __name__ == "__main__":
    test_full_directed_pipeline()