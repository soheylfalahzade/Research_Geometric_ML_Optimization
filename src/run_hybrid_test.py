import networkx as nx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SpannerOptimizer:
    def __init__(self, t=2.0):
        self.t = t
        # Use the best parameters found by the genetic algorithm
        self.model = RandomForestClassifier(
            n_estimators=127, 
            max_depth=7, 
            min_samples_split=3, 
            class_weight='balanced',
            random_state=42
        )
        
    def repair(self, G, pruned_G):
        """Repair algorithm: restores the critical edges that violate the t condition"""
        repaired_G = pruned_G.copy()
        violations = 0
        
        # Extract the edges the model removed (Potential missing critical edges)
        nodes = list(G.nodes())
        print("Starting Repair Module: Scanning for connectivity gaps...")
        
        # Random test to evaluate the stretch factor at large scale
        for i in range(1000): 
            u, v = np.random.choice(nodes, 2, replace=False)
            
            # If a path exists in the original graph, it must also exist in the pruned graph
            try:
                d_orig = nx.shortest_path_length(G, u, v, weight='length')
                try:
                    d_pruned = nx.shortest_path_length(repaired_G, u, v, weight='length')
                    if d_pruned > self.t * d_orig:
                        # t condition violated: restore the direct edge to bring back the stretch bound
                        if G.has_edge(u, v):
                            repaired_G.add_edge(u, v, length=G[u][v]['length'])
                            violations += 1
                except nx.NetworkXNoPath:
                    # the graph became disconnected: restore the edge to keep connectivity
                    if G.has_edge(u, v):
                        repaired_G.add_edge(u, v, length=G[u][v]['length'])
                        violations += 1
            except nx.NetworkXNoPath:
                continue
                
            if i % 200 == 0:
                print(f"Repair Progress: {i}/1000 samples checked...")
                
        return repaired_G, violations

    def optimize(self, G, df):
        # Step 1: smart pruning with ML
        features = ["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]
        X = df[features]
        df['predicted'] = self.model.predict(X)
        
        # Build the initial pruned graph
        pruned_G = nx.Graph()
        pruned_G.add_nodes_from(G.nodes())
        # Only the edges the model said 1 (keep)
        for _, row in df[df['predicted'] == 1].iterrows():
            pruned_G.add_edge(int(row['node_u']), int(row['node_v']), length=row['length'])
            
        # Step 2: repair (enforcing the per-edge stretch bound)
        final_G, total_repairs = self.repair(G, pruned_G)
        return final_G, total_repairs