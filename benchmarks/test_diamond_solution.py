import os
import time
import torch
import numpy as np
import networkx as nx
import osmnx as ox
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# وارد کردن مدل و توابع پایه‌ای از فایل اصلی
from spanner_pipeline import GeometricEdgeSAGE, GLV_T_LIMIT, NUM_DIJKSTRA_THREADS, _check_one_edge

def glv_repair_batched(G_sparse, removed_edges_info, t_limit=GLV_T_LIMIT, batch_size=250):
    """
    راهکار پیشرفته داور: ترمیم دسته‌ای پویا (Batch-Sequential Repair)
    یال‌ها را در دسته‌های کوچک‌تر بررسی کرده و با پر کردن گراف، مانع ترمیم‌های تکراری و بیهوده می‌شود.
    """
    repaired_G = G_sparse.copy()
    repairs    = []

    # مرتب‌سازی: کوتاه‌ترین لبه‌ها اول
    candidates = sorted(removed_edges_info, key=lambda e: e["length"])
    num_candidates = len(candidates)
    
    # تقسیم کاندیداها به دسته‌های batch_size تایی
    for start_idx in range(0, num_candidates, batch_size):
        batch = candidates[start_idx : start_idx + batch_size]
        
        # بررسی موازی اعضای این دسته روی آخرین وضعیت گراف ترمیم‌شده تا این لحظه
        args_list = [(repaired_G, edge, t_limit) for edge in batch]
        batch_needs_repair = {}
        
        with ThreadPoolExecutor(max_workers=NUM_DIJKSTRA_THREADS) as pool:
            for idx, needs in pool.map(_check_one_edge, args_list):
                batch_needs_repair[idx] = needs
                
        # اعمالِ sequential و بلادرنگ یال‌های بحرانی این دسته به گراف
        # این کار باعث می‌شود در دسته‌های بعدی، این یال‌ها به عنوان مسیرهای جایگزین دیده شوند.
        for edge in batch:
            if batch_needs_repair.get(edge["idx"], False):
                repaired_G.add_edge(edge["u"], edge["v"], length=edge["length"])
                repairs.append(edge["idx"])
                
    return repaired_G, repairs

def test_single_city_diamond(city_label="Eindhoven", city_query="Eindhoven, Netherlands"):
    print(f"\n--- Testing Diamond Solution on [{city_label}] ---")
    weights_path = 'best_base_model.pt'
    
    # بارگذاری مدل
    model = GeometricEdgeSAGE()
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    model.eval()
    
    # دانلود گراف
    print(f"[{city_label}] Downloading OSM graph...")
    G = ox.graph_from_place(city_query, network_type="drive")
    G = ox.project_graph(G).to_undirected()
    
    # ۱. راهکار علمی اول داور: شناسایی و قفل کردن پل‌ها (Bridges)
    print(f"[{city_label}] Identifying topological bridges (O(V+E) time)...")
    t_start_bridges = time.time()
    bridges = {tuple(sorted((u, v))) for u, v in nx.bridges(G)}
    print(f"[{city_label}] Found {len(bridges)} critical bridge edges in {time.time() - t_start_bridges:.2f}s.")
    
    # استخراج ویژگی‌ها و استنتاج مدل
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
    
    with torch.no_grad():
        probs = torch.sigmoid(model(x_local, edge_idx, edge_attr_loc)).numpy().flatten()
        
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes())
    removed_edges = []
    
    for i, (u, v, k, d) in enumerate(edge_list):
        edge_key = tuple(sorted((u, v)))
        is_bridge = edge_key in bridges
        
        # اگر یال خروجی مدل بالایی دارد، یا یک "پل" است، آن را قفل کرده و هرگز هرس نمی‌کنیم
        if probs[i] > 0.55 or is_bridge:
            G_sparse.add_edge(u, v, length=d["length"])
        else:
            removed_edges.append({
                "u": u, "v": v, "length": d["length"], "prob": probs[i], "idx": i
            })
            
    # ۲. اعمال ترمیم دسته‌ای موازی-ترتیبی
    print(f"[{city_label}] Running Batch-Sequential GLV-Repair ({len(removed_edges)} candidates)...")
    t0 = time.time()
    G_final, repairs = glv_repair_batched(G_sparse, removed_edges, batch_size=250)
    elapsed = time.time() - t0
    
    # محاسبه متریک‌ها با استفاده از SciPy دایجسترا فوق‌سریع روی ۱۰۰,۰۰۰ نمونه تصادفی
    from spanner_pipeline import compute_global_stretch_scipy
    print(f"[{city_label}] Computing global stretch (10^5 pairs)...")
    stretches = compute_global_stretch_scipy(G, G_final, num_samples=100000)
    
    sparsification = (1 - G_final.number_of_edges() / G.number_of_edges()) * 100
    mean_stretch = np.mean(stretches)
    max_stretch = np.max(stretches)
    
    print("\n" + "="*50)
    print(f"DIAMOND TEST RESULTS FOR {city_label.upper()}")
    print("="*50)
    print(f"Sparsification:  {sparsification:.2f}%")
    print(f"Avg Stretch:     {mean_stretch:.4f}")
    print(f"Max Stretch:     {max_stretch:.4f}")
    print(f"Repairs Needed:  {len(repairs)} (out of {len(removed_edges)} pruned)")
    print(f"Repair Time:     {elapsed:.2f}s")
    print("="*50)

if __name__ == "__main__":
    test_single_city_diamond()