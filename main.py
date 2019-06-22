import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import numpy as np
import metis

# initiate global variable here
G = nx.DiGraph()

# Create node
with open("datasets/nodes.csv", "r") as f:
    for line in f:
        N = int(line.split(' ')[0])
        X = float(line.split(' ')[1])
        Y = float(line.split(' ')[2])
        G.add_node(N, pos=(X,Y))

# Create edge
with open("datasets/edges.csv", "r") as f:
    for line in f:
        U = int(line.split(' ')[1])
        V = int(line.split(' ')[2])
        W = float(line.split(' ')[3])
        G.add_edge(U, V, weight=W)

# Get adjacency-matrix as numpy-array
# adj_mat = nx.to_numpy_matrix(G)

# Cluster
# sc = SpectralClustering(2, affinity='precomputed', n_init=100)
# sc.fit(adj_mat)
(edgecuts, parts) = metis.part_graph(G, 3)
colors = ['red','blue','green']
for i, p in enumerate(parts):
    G.node[i]['color'] = colors[p]

# Compare ground-truth and clustering-results
# print('spectral clustering')
# print(sc.labels_)
# print('just for better-visualization: invert clusters (permutation)')
# print(np.abs(sc.labels_ - 1))

# Drawing Code
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_size=50)
# nx.draw_networkx_labels(G, pos, font_size=20, font_color='white', font_family='sans-serif')
# plt.savefig("graph.png")
plt.show()