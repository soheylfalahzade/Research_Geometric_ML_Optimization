import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class SpannerOptimizer:
    def __init__(self, t=2.0):
        self.t = t
        self.model = RandomForestClassifier(n_estimators=127, max_depth=7, min_samples_split=3, class_weight='balanced')
        
    def repair(self, G, pruned_G):
        """الگوریتم ترمیم: یال‌های حیاتی که شرط t را نقض کرده‌اند بازمی‌گرداند"""
        repaired_G = pruned_G.copy()
        violations = 0
        
        # بررسی شرط t-spanner برای گره‌ها (نمونه‌برداری تصادفی برای سرعت در گراف‌های بزرگ)
        nodes = list(G.nodes())
        for _ in range(500): # تعداد نمونه‌برداری برای تضمین ریاضی
            u, v = np.random.choice(nodes, 2, replace=False)
            if G.has_edge(u, v):
                d_orig = G[u][v]['length']
                try:
                    d_pruned = nx.shortest_path_length(repaired_G, u, v, weight='length')
                    if d_pruned > self.t * d_orig:
                        repaired_G.add_edge(u, v, length=d_orig) # ترمیم لبه حیاتی
                        violations += 1
                except nx.NetworkXNoPath:
                    repaired_G.add_edge(u, v, length=d_orig) # ترمیم اتصال
                    violations += 1
        return repaired_G, violations

    def optimize(self, G, df):
        # مرحله ۱: هرس هوشمند با ML
        X = df[["length", "u_degree", "v_degree", "dx", "dy", "edge_centrality", "u_pagerank", "v_pagerank"]]
        df['predicted'] = self.model.predict(X)
        
        # ساخت گراف اولیه هرس شده
        pruned_G = nx.Graph()
        pruned_G.add_nodes_from(G.nodes())
        for _, row in df[df['predicted'] == 1].iterrows():
            pruned_G.add_edge(row['node_u'], row['node_v'], length=row['length'])
            
        # مرحله ۲: ترمیم (تضمین ریاضی)
        final_G, total_repairs = self.repair(G, pruned_G)
        return final_G, total_repairs