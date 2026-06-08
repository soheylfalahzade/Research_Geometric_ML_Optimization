import pandas as pd
import numpy as np
import networkx as nx

def compute_greedy_spanner(points, t=2.0):
    n = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            edges.append((i, j, dist))
    edges.sort(key=lambda x: x[2])
    for u, v, weight in edges:
        try:
            shortest_path = nx.shortest_path_length(G, source=u, target=v, weight='weight')
        except nx.NetworkXNoPath:
            shortest_path = float('inf')
        if shortest_path > t * weight:
            G.add_edge(u, v, weight=weight)
    return G

def generate_pro_research_data(num_graphs=100, nodes_per_graph=30):
    print(f"🧬 Extracting ELITE Geometric Features (Length, Rank, Angle) from {num_graphs} graphs...")
    all_data = []

    for g_idx in range(num_graphs):
        points = np.random.rand(nodes_per_graph, 2) * 100
        spanner = compute_greedy_spanner(points, t=2.0)
        
        for i in range(nodes_per_graph):
            neighbors = []
            for j in range(nodes_per_graph):
                if i == j: continue
                dist = np.linalg.norm(points[i] - points[j])
                angle = np.arctan2(points[j][1] - points[i][1], points[j][0] - points[i][0])
                neighbors.append({'id': j, 'dist': dist, 'angle': angle})
            
            # مرتب‌سازی بر اساس فاصله
            neighbors.sort(key=lambda x: x['dist'])
            
            for rank, neighbor in enumerate(neighbors):
                j = neighbor['id']
                if i < j:
                    # محاسبه Angular Gap: کمترین اختلاف زاویه با یال‌های کوتاه‌تر
                    angles_of_shorter_edges = [n['angle'] for n in neighbors[:rank]]
                    if not angles_of_shorter_edges:
                        min_angle_gap = 2 * np.pi # برای اولین یال
                    else:
                        angle_diffs = [abs(neighbor['angle'] - a) for a in angles_of_shorter_edges]
                        min_angle_gap = min(angle_diffs)

                    all_data.append({
                        'edge_length': neighbor['dist'],
                        'relative_rank': rank,
                        'angular_gap': min_angle_gap, # ویژگی فوق‌حرفه‌ای جدید
                        'is_in_spanner': 1 if spanner.has_edge(i, j) else 0
                    })
        
        if (g_idx + 1) % 20 == 0:
            print(f"✅ Step {g_idx + 1}/{num_graphs} complete.")

    df = pd.DataFrame(all_data)
    df.to_csv("spanner_dataset_pro.csv", index=False)
    print("\n🏁 ELITE Dataset saved as 'spanner_dataset_pro.csv'")

if __name__ == "__main__":
    generate_pro_research_data(num_graphs=150, nodes_per_graph=40)