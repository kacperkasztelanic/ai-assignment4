import numpy as np
from scipy.spatial import distance


def find_pairs_euclidean(l1, l2):
    v1 = np.array([x.vector for x in l1])
    v2 = np.array([x.vector for x in l2])
    distances = distance.cdist(v1, v2, metric='euclidean')
    pairs = []
    for i in range(len(l1)):
        corresponding = np.argmin(distances[i, :])
        if (i == np.argmin(distances[:, corresponding])):
            pairs.append((l1[i], l2[corresponding]))
    return pairs


def find_nearest_neighbors(points, n):
    v = np.array([x.coords for x in points])
    distances = distance.cdist(v, v, metric='euclidean')
    l = []
    for i in range(len(points)):
        closest = points[np.argsort(distances[i])[1:(n + 1)]]
        l.append((points[i], closest))
        print(closest)
    return l


def filter_points(pairs, n, threshold):
    points1 = np.array([x[0] for x in pairs])
    points2 = np.array([x[1] for x in pairs])
    l1 = find_nearest_neighbors(points1, n)
    l2 = find_nearest_neighbors(points2, n)
    filtered_pairs = []
    for i in range(len(points1)):
        for j in range(n):
            if(l1[i][1][j].id == pairs[i][1]):
                pass

