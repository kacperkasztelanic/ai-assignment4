import numpy as np
from scipy.spatial import distance
import sys


def corresponding_points(l1, l2):
    distances = distance.cdist(
        np.array([point.vector for point in l1]),
        np.array([point.vector for point in l2]), metric='euclidean')
    pairs = []
    for i in range(len(l1)):
        closest_column = np.argmin(distances[i])
        if i == np.argmin(distances[:, closest_column]):
            pairs.append([l1[i], l2[closest_column]])
    return np.array(pairs)


def filter_pairs(pairs, n, threshold):
    neighbors1 = n_closest_points(pairs[:, 0], n)
    neighbors2 = n_closest_points(pairs[:, 1], n)
    consistency = corresponding_coverage(neighbors1, neighbors2)
    return pairs[np.argwhere(consistency > threshold).flatten()]


def n_closest_points(points, n):
    v = np.array([p.coords for p in points])
    dist = distance.cdist(v, v, metric='euclidean')
    return np.argsort(dist + np.eye(dist.shape[0]) * sys.maxsize, axis=1)[:, :n]


def corresponding_coverage(neighbors1, neighbors2):
    col_size = neighbors1.shape[0]
    result = np.zeros(col_size)
    for i in range(col_size):
        result[i] = np.intersect1d(neighbors1[i], neighbors2[i]).shape[0]
    result /= neighbors1.shape[1]
    return result
