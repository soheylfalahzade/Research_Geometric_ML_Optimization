import osmnx as ox
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
import time

def test_directed_logic_and_scc():
    print("--- 🚦 Initiating Directed Graph & SCC Validation Test ---")
    city_name = "Eindhoven, Netherlands"
    
    # 1. Load the graph fully directed (without to_undirected)
    print(f"[Input] Downloading directed graph for {city_name}...")
    G_directed = ox.graph_from_place(city_name, network_type='drive')
    # Unlike earlier runs, we no longer make the graph undirected!
    
    num_nodes = G_directed.number_of_nodes()
    num_edges = G_directed.number_of_edges()
    print(f"[Stat] Nodes: {num_nodes}, Directed Edges: {num_edges}")

    # 2. Convert to a directed sparse matrix (asymmetric)
    # Here the edge A->B is not necessarily equal to B->A
    nodes = list(G_directed.nodes())
    matrix_dir = nx.adjacency_matrix(G_directed, nodelist=nodes, weight='length')

    # 3. Compute Dijkstra in a directed manner (Directed=True)
    print("[Dijkstra] Running directed shortest path calculations (SciPy)...")
    t0 = time.time()
    # Select 100 random nodes to test speed and correctness
    rng = np.random.default_rng(42)
    sources = rng.choice(num_nodes, size=100, replace=False)
    
    # Note: directed=True means move only in the direction of the street arrows
    dist_matrix = dijkstra(matrix_dir, directed=True, indices=sources)
    
    print(f"[Dijkstra] Finished in {time.time() - t0:.4f}s")

    # 4. Check strong connectivity (Strongly Connected Components)
    # This is the core of the claim being tested
    print("[SCC] Verifying Strong Connectivity...")
    n_components, labels = connected_components(csgraph=matrix_dir, directed=True, connection='strong')
    
    # In an ideal road graph there should be only 1 large component
    # If n_components > 1, parts of the city are disconnected from each other
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