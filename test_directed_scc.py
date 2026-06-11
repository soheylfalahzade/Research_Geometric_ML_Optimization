import osmnx as ox
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
import time

def test_directed_logic_and_scc():
    print("--- 🚦 Initiating Directed Graph & SCC Validation Test ---")
    city_name = "Eindhoven, Netherlands"
    
    # ۱. لود کردن گراف به صورت کاملاً جهت‌دار (بدون to_undirected)
    print(f"[Input] Downloading directed graph for {city_name}...")
    G_directed = ox.graph_from_place(city_name, network_type='drive')
    # برخلاف دفعات قبل، ما دیگر گراف را Undirected نمی‌کنیم!
    
    num_nodes = G_directed.number_of_nodes()
    num_edges = G_directed.number_of_edges()
    print(f"[Stat] Nodes: {num_nodes}, Directed Edges: {num_edges}")

    # ۲. تبدیل به ماتریس اسپرس جهت‌دار (نامتقارن)
    # در اینجا یال A->B لزوماً با B->A برابر نیست
    nodes = list(G_directed.nodes())
    matrix_dir = nx.adjacency_matrix(G_directed, nodelist=nodes, weight='length')

    # ۳. محاسبه دایجسترا به صورت جهت‌دار (Directed=True)
    print("[Dijkstra] Running directed shortest path calculations (SciPy)...")
    t0 = time.time()
    # انتخاب ۱۰۰ گره تصادفی برای تست سرعت و صحت
    rng = np.random.default_rng(42)
    sources = rng.choice(num_nodes, size=100, replace=False)
    
    # توجه: directed=True یعنی فقط در جهت فلش خیابان‌ها حرکت کن
    dist_matrix = dijkstra(matrix_dir, directed=True, indices=sources)
    
    print(f"[Dijkstra] Finished in {time.time() - t0:.4f}s")

    # ۴. بررسی اتصال قوی (Strongly Connected Components)
    # این قلب تپنده ادعای علمی ماست
    print("[SCC] Verifying Strong Connectivity...")
    n_components, labels = connected_components(csgraph=matrix_dir, directed=True, connection='strong')
    
    # در یک گراف حمل‌ونقل ایده‌آل، باید فقط ۱ مولفه بزرگ داشته باشیم
    # اگر n_components > 1 باشد، یعنی بخش‌هایی از شهر از هم جدا هستند
    main_component_size = np.max(np.bincount(labels))
    connectivity_ratio = (main_component_size / num_nodes) * 100

    print("\n" + "="*50)
    print("      DIRECTED LOGIC TEST RESULTS")
    print("="*50)
    print(f"Matrix Symmetry: {'Symmetric' if (matrix_dir != matrix_dir.T).nnz == 0 else 'Asymmetric (Correct for Directed)'}")
    print(f"Number of SCCs:  {n_components}")
    print(f"Main SCC Ratio:  {connectivity_ratio:.2f}% of nodes reachable")
    print(f"Directed Dijkstra Mean Dist: {np.mean(dist_matrix[dist_matrix < np.inf]):.2f} meters")
    print("="*50)
    
    if connectivity_ratio > 90:
        print("\n✅ SUCCESS: Directed logic is sound. We can now enforce SCC=100% in the main pipeline.")
    else:
        print("\n⚠️ WARNING: Graph has isolated components. We must use the 'Main SCC' as our base graph.")

if __name__ == "__main__":
    test_directed_logic_and_scc()