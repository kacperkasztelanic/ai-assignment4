import numpy as np
from scipy.spatial import distance
import sys


def corresponding_points(l1, l2):
    """
    For each pair of FeaturedPoint finds euclidean distance of theirs vector, and returns pairs of points that are
    closest to each others
    :param l1: points from image 1
    :param l2: points from image 2
    :return:
    """
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
    """
    Filter pairs based on n and threshold
    :param pairs: vector of pairs
    :param n: Quantity of closest points
    :param threshold: value in range 0.0 - 1.0
    :return: filtered vector of pairs
    """
    neighbors1 = n_closest_points(pairs[:, 0], n)
    neighbors2 = n_closest_points(pairs[:, 1], n)
    consistency = corresponding_coverage(neighbors1, neighbors2)
    return pairs[np.argwhere(consistency > threshold).flatten()]


def n_closest_points(points, n):
    """
    For set of a points finds Euclidean distance on Cartesian coordinate system for each permutation of points
    Based on distance matrix finds n closest points for each points
    :param points: set of Points 1xN
    :param n: Quantity of closest points
    :return: MxN
    """
    v = np.array([p.coords for p in points])
    dist = distance.cdist(v, v, metric='euclidean')
    return np.argsort(dist + np.eye(dist.shape[0]) * sys.maxsize, axis=1)[:, :n]


def corresponding_coverage(neighbors1, neighbors2):
    """
    For corresponding points finds percentage how many closest points are matching on both images
    :param neighbors1: matrix MxN
    :param neighbors2: matrix MxN
    :return: vector M contains values in range 0.0 - 1.0
    """
    col_size = neighbors1.shape[0]
    result = np.zeros(col_size)
    for i in range(col_size):
        result[i] = np.intersect1d(neighbors1[i], neighbors2[i]).shape[0]
    result /= neighbors1.shape[1]
    return result
