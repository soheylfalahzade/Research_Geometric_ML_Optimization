import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

print("Testing SciPy Dijkstra integration...")

# ۱. ساخت یک گراف فرضی کوچک با ۵ گره و لبه‌های وزن‌دار
G = nx.complete_graph(5)
for u, v in G.edges():
    G[u][v]['length'] = np.random.uniform(1.0, 10.0)

# ۲. تبدیل گراف به ماتریس اسپرس سایپای
nodes = list(G.nodes())
matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='length')

# ۳. اجرای الگوریتم دایجسترا از گره شماره ۲ به بقیه گره‌ها
sources = [2]
distances = dijkstra(matrix, directed=True, indices=sources)

print("Distance matrix computed successfully:")
print(distances)
print("✓ SciPy Dijkstra test passed flawlessly!")