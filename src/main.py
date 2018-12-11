from math import sqrt
from copy import deepcopy
import numpy as np

import networkx as nx

from plot import Plotter

np.set_printoptions(precision=3)


def mu(row, n):
    """Computes the means for every row. This function is passed as an
    argument to map.

    """
    return len(row[1]) / (n - 1)


def sigma(adj, means):
    """Computes the variance for a whole 2D array

    """
    variances = []
    for i, row in enumerate(adj):
        res = 0
        for j, _ in row[1].items():
            res += (1 - means[i])**2
        variances.append(sqrt(res))
    return variances


def p_correlation(adj, means, variances):
    """Returns a 2D matrix with the pearson correlation coefficient for
    each element in arr

    """
    simil = np.ones((len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if i == j:
                simil[i][j] = np.nan
            else:
                acc = 0
                adj_node_i = adj[i][1]
                adj_node_j = adj[j][1]
                for k in range(len(adj)):
                    neighbour_i = 1 if k in adj_node_i else 0
                    neighbour_j = 1 if k in adj_node_j else 0
                    acc += (neighbour_i - means[i]) * (neighbour_j - means[j])
                try:
                    simil[i, j] = ((1/(len(adj) - 3)) * acc) / (variances[i] * variances[j])
                except ZeroDivisionError:
                    simil[i, j] = 0
    return simil


def slink(sim):
    """Using single-linkage, builds a tree of clusters, where at each
    level (index of array tree)

    """
    tree = [{}]
    for x in range(len(sim) - 1):
        max_ij = np.unravel_index(np.nanargmax(sim), (len(sim), len(sim)))
        keep = min(max_ij)
        throw = max(max_ij)
        for i in range(len(sim)):
            if keep != i and throw != i:
                sim[keep, i] = max(sim[keep, i], sim[throw, i])
                sim[i, keep] = max(sim[i, keep], sim[i, throw])
            if throw != i:
                sim[throw, i] = np.NINF
                sim[i, throw] = np.NINF

        before = tree[len(tree) - 1]
        tree.append(deepcopy(before))
        now = tree[len(tree) - 1]
        if keep in now and throw in now:
            now[keep].extend(now[throw])
            now[keep].append(throw)
            now.pop(throw)
        elif keep in now:
            now[keep].append(throw)
        elif throw in now:
            now[keep] = [throw]
            now[keep].extend(now[throw])
            now.pop(throw)
        else:
            now[keep] = [throw]

    return tree


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
