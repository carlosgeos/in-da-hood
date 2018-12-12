from link import link
from measures import mu, sigma, p_correlation
from plot import Plotter
from copy import deepcopy
import numpy as np
import networkx as nx

np.set_printoptions(precision=3)


# Graph definition
n = 200
G = nx.random_geometric_graph(n, 0.1)
adj = list(G.adjacency())

# Derived structural information
means = list(map(lambda x: mu(x, n), adj))
variances = sigma(adj, means)
similarities = p_correlation(adj, means, variances)

# Hierarchical clustering
stree = link(deepcopy(similarities), max)
ctree = link(deepcopy(similarities), min)

splotter = Plotter(G, stree, "slink.html")
cplotter = Plotter(G, ctree, "clink.html")

level1 = n - 40
splotter.plot(level1)
cplotter.plot(level1)

level2 = n - 20
splotter.plot(level2)
cplotter.plot(level2)

if __name__ == '__main__':
    pass
