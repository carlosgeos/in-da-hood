from copy import deepcopy
import numpy as np


def clink(sim):
    """Using complete-linkage, builds a tree of clusters, where at each
    level (index of array tree), we can find several of them

    """
    tree = [{}]
    for x in range(len(sim) - 1):
        max_ij = np.unravel_index(np.nanargmax(sim), (len(sim), len(sim)))
        print("MaxIJ is:", max_ij)
        keep = min(max_ij)
        throw = max(max_ij)
        for i in range(len(sim)):
            if keep != i and throw != i:
                sim[keep, i] = min(sim[keep, i], sim[throw, i])
                sim[i, keep] = min(sim[i, keep], sim[i, throw])
            if throw != i:
                sim[throw, i] = np.NINF
                sim[i, throw] = np.NINF

        # Procedure to iteratively build the tree
        before = tree[len(tree) - 1]
        tree.append(deepcopy(before))
        print(before)
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
