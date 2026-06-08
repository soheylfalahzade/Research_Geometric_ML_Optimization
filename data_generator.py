import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np

def generate_real_spatial_dataset(city_name="Eindhoven, Netherlands", stretch_factor=2.0):
    print(f"Loading real road network for {city_name}...")
    G = ox.graph_from_place(city_name, network_type="drive")
    G = ox.project_graph(G)
    G_undirected = G.to_undirected()
    
    print("Extracting node coordinates and global centrality metrics...")
    nodes, edges = ox.graph_to_gdfs(G_undirected)
    
    # Calculate PageRank as a global topological feature
    print("Calculating PageRank for global context...")
    pagerank = nx.pagerank(G_undirected, weight='length')
    
    # Sort edges by length for Greedy Spanner
    print("Sorting edges for Greedy Spanner computation...")
    sorted_edges = sorted(G_undirected.edges(keys=True, data=True), key=lambda x: x[3].get('length', 0.0))
    
    spanner_G = nx.Graph()
    spanner_G.add_nodes_from(G_undirected.nodes())
    
    spanner_edge_set = set()
    total_edges = len(sorted_edges)
    print(f"Computing EXACT Greedy Spanner (t={stretch_factor}) on {total_edges} edges...")
    
    for idx, (u, v, k, data) in enumerate(sorted_edges):
        w = data.get('length', 1.0)
        try:
            shortest_path_len = nx.shortest_path_length(spanner_G, source=u, target=v, weight='length')
        except nx.NetworkXNoPath:
            shortest_path_len = float('inf')
            
        if shortest_path_len > stretch_factor * w:
            spanner_G.add_edge(u, v, length=w)
            spanner_edge_set.add((u, v, k))
            
        if idx % 2000 == 0:
            print(f"Progress: {idx}/{total_edges} edges evaluated...")

    print("Extracting geometric and topological features for ML dataset...")
    edges_data = []
    for u, v, k, data in G_undirected.edges(keys=True, data=True):
        length = data.get("length", 0.0)
        u_degree = G_undirected.degree(u)
        v_degree = G_undirected.degree(v)
        
        # Coordinate calculation
        u_x, u_y = nodes.loc[u]['x'], nodes.loc[u]['y']
        v_x, v_y = nodes.loc[v]['x'], nodes.loc[v]['y']
        dx = v_x - u_x
        dy = v_y - u_y
        
        edge_centrality = u_degree * v_degree / (length + 1)
        
        # Global topological features
        u_pagerank = pagerank.get(u, 0.0)
        v_pagerank = pagerank.get(v, 0.0)
        
        is_spanner_edge = 1 if (u, v, k) in spanner_edge_set or (v, u, k) in spanner_edge_set else 0
        
        edges_data.append({
            "node_u": u,
            "node_v": v,
            "length": length,
            "u_degree": u_degree,
            "v_degree": v_degree,
            "dx": dx,
            "dy": dy,
            "edge_centrality": edge_centrality,
            "u_pagerank": u_pagerank,
            "v_pagerank": v_pagerank,
            "is_spanner_edge": is_spanner_edge
        })
        
    df = pd.DataFrame(edges_data)
    df.to_csv("spanner_dataset_pro.csv", index=False)
    print(f"Successfully generated REAL spatial dataset with PageRank: {len(df)} edges saved.")

if __name__ == "__main__":
    generate_real_spatial_dataset()