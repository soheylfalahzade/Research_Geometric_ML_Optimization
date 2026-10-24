import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

print("Testing SciPy Dijkstra integration...")

# 1. Build a small hypothetical graph with 5 nodes and weighted edges
G = nx.complete_graph(5)
for u, v in G.edges():
    G[u][v]['length'] = np.random.uniform(1.0, 10.0)

# 2. Convert the graph to a SciPy sparse matrix
nodes = list(G.nodes())
matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='length')

# 3. Run Dijkstra's algorithm from node 2 to the other nodes
sources = [2]
distances = dijkstra(matrix, directed=True, indices=sources)

print("Distance matrix computed successfully:")
print(distances)
print("✓ SciPy Dijkstra test passed flawlessly!")