from slink import slink
from measures import mu, sigma, p_correlation
from plot import Plotter
import numpy as np
import networkx as nx

np.set_printoptions(precision=3)


# Graph definition
n = 100
G = nx.random_geometric_graph(n, 0.1)
adj = list(G.adjacency())

# Derived structural information
means = list(map(lambda x: mu(x, n), adj))
variances = sigma(adj, means)
similarities = p_correlation(adj, means, variances)

# Hierarchical clustering
tree = slink(similarities)
level = int(n - n/5)

plotter = Plotter(G, tree[level])

plotter.plot()

if __name__ == '__main__':
    pass
