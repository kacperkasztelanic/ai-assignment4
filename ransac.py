import numpy as np
from scipy.spatial import distance
from Point import Point

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def affine_array(x1, y1, x2, y2, x3, y3, u1, v1, u2, v2, u3, v3):
    a1 = np.array([[x1, y1, 1, 0, 0, 0],
                   [x2, y2, 1, 0, 0, 0],
                   [x3, y3, 1, 0, 0, 0],
                   [0, 0, 0, x1, y1, 1],
                   [0, 0, 0, x2, y2, 1],
                   [0, 0, 0, x3, y3, 1]])
    a2 = np.array([u1, u2, u3, v1, v2, v3])
    if not is_invertible(a1):
        return None
    a = np.linalg.inv(a1) @ a2
    res = np.reshape(np.append(a, [0, 0, 1]), newshape=(3, 3))
    return res


def perspective_array(x1, y1, x2, y2, x3, y3, x4, y4, u1, v1, u2, v2, u3, v3, u4, v4):
    a1 = np.array([[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1],
                   [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2],
                   [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3],
                   [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4],
                   [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1],
                   [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2],
                   [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3],
                   [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4]])
    a2 = np.array([u1, u2, u3, u4, v1, v2, v3, v4])
    if not is_invertible(a1):
        return None
    a = np.linalg.inv(a1) @ a2
    a = np.append(a, 1)
    res = np.reshape(a, newshape=(3, 3))
    return res


def calc_model(samples):
    x1 = samples[0][0].coords[0]
    y1 = samples[0][0].coords[1]
    x2 = samples[1][0].coords[0]
    y2 = samples[1][0].coords[1]
    x3 = samples[2][0].coords[0]
    y3 = samples[2][0].coords[1]
    u1 = samples[0][1].coords[0]
    v1 = samples[0][1].coords[1]
    u2 = samples[1][1].coords[0]
    v2 = samples[1][1].coords[1]
    u3 = samples[2][1].coords[0]
    v3 = samples[2][1].coords[1]
    if len(samples) == 3:
        a = affine_array(x1, y1, x2, y2, x3, y3, u1, v1, u2, v2, u3, v3)
    else:
        x4 = samples[3][0].coords[0]
        y4 = samples[3][0].coords[1]
        u4 = samples[3][1].coords[0]
        v4 = samples[3][1].coords[1]
        a = perspective_array(x1, y1, x2, y2, x3, y3, x4, y4, u1, v1, u2, v2, u3, v3, u4, v4)
    return a


def model_error(model, pair):
    return distance.cdist(np.reshape(model @ np.array([pair[0].coords[0], pair[0].coords[1], 1]), newshape=(-1, 1)),
                          np.array([pair[1].coords[0], pair[1].coords[1], 1]).reshape(-1, 1), metric='euclidean').flatten()[0]


def ransac_model(pairs, s, iter, max_error):
    best_model = None
    best_score = 0
    for i in range(iter):
        model = None
        while model is None:
            indices = np.random.choice(pairs.shape[0], size=s)
            chosen = pairs[indices]
            model = calc_model(chosen)
        score = 0
        for pair in pairs:
            error = model_error(model, pair)
            if (error < max_error):
                score += 1
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def ransac_pairs(pairs, model):
    transformed_pairs = []
    for pair in pairs:
        temp = model @ np.array([pair[0].coords[0], pair[0].coords[1], 1])
        transformed_pairs.append((pair[0], Point((temp[0], temp[1]), pair[1].vector)))
    return transformed_pairs

