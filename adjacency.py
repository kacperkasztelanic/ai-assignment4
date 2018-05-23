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
    list_ = []
    for i in range(len(points)):
        closest = points[np.argsort(distances[i])[1:(n + 1)]]
        list_.append((points[i], closest))
        # print(closest)
    return list_


def filter_points(pairs, n, threshold):
    # print(pairs)
    points1 = np.array([x[0] for x in pairs])
    points2 = np.array([x[1] for x in pairs])
    l1 = find_nearest_neighbors(points1, n)
    # print(l1)
    l2 = find_nearest_neighbors(points2, n)
    filtered_pairs = []
    for i in range(len(points1)):
        for j in range(n):
            if(l1[i][1][j].id == pairs[i][1]):
                pass






def filter_pairs(pairs, n, threshold):
    points1 = np.array([x[0] for x in pairs])
    points2 = np.array([x[1] for x in pairs])
    dist1 = generate_dist(points1)
    dist2 = generate_dist(points2)
    neightbours1 = get_n_smallest(dist1, n)
    neightbours2 = get_n_smallest(dist2, n)
    consistancy = get_equivalens_size(neightbours1, neightbours2)
    indexes = selection(consistancy, threshold)
    # print(indexes)
    return np.asarray(pairs)[indexes]
    # print(pairs)

    # print(dist1.shape)
    # print(neightbours1)
    # print(neightbours2)
    # print(consistancy)


def generate_dist(points):
    v = np.array([p.coords for p in points])
    return distance.cdist(v, v, metric='euclidean')


def get_n_smallest(dist, n):
    temp = np.argsort(dist, axis=1)[:, 0: n + 1]
    col_size = dist.shape[0]
    result = temp[:, 0:n]
    for i in range(col_size):
        for j in range(n):
            if result[i, j] == i:
                result[i, j] = temp[i, n]
    return result


def get_equivalens_size(neightbours1, neightbours2):
    col_size = neightbours1.shape[0]
    result = np.zeros(col_size)
    for i in range(col_size):
        result[i] = np.intersect1d(neightbours1[i], neightbours2[i]).shape[0]
    result /= neightbours1.shape[1]
    return result

def selection(consistancy, threshold):
    return np.argwhere(consistancy > threshold).flatten()