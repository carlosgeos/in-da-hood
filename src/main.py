from math import sqrt
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
    tree = [{}]
    for x in range(len(sim) - 1):
        max_cell = np.nanmax(sim)
        print("max_value is:", max_cell)
        max_ij = np.unravel_index(np.nanargmax(sim), (len(sim), len(sim)))
        print("maxij is: ", max_ij)
        keep = min(max_ij)
        throw = max(max_ij)
        for i in range(len(sim)):
            if keep != i and throw != i:
                sim[keep][i] = max(sim[keep][i], sim[throw][i])
                sim[i][keep] = max(sim[i][keep], sim[i][throw])
        sim = np.delete(sim, throw, 0)
        sim = np.delete(sim, throw, 1)
        before = tree[x]
        tree.append(dict(before))
        if keep:
            pass






n = 100
G = nx.random_geometric_graph(n, 0.3)
adj = list(G.adjacency())

means = list(map(lambda x: mu(x, n), adj))
variances = sigma(adj, means)
similarities = p_correlation(adj, means, variances)

print("Means:", means)
print("Variances:", variances)
print("Similarities:\n", similarities)

plotter = Plotter(G)

plotter.plot()

slink(similarities)

if __name__ == '__main__':
    pass
