import numpy as np
from scipy.spatial import distance


def find_pairs_euclidean(l1, l2, threshold):
    v1 = np.array([x.vector for x in l1])
    v2 = np.array([x.vector for x in l2])
    distances = distance.cdist(v1, v2, metric='euclidean')
    pairs = []
    for i in range(len(l1)):
        corresponding = np.argmin(distances[i, :])
        if (i == np.argmin(distances[:, corresponding]) and distances[i, corresponding] < threshold):
            pairs.append((l1[i], l2[corresponding]))
    return pairs
