import numpy as np
from scipy.spatial import distance


def create_array(samples):
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

    a1 = np.array([[x1, y1, 1, 0, 0, 0],
                   [x2, y2, 1, 0, 0, 0],
                   [x3, y3, 1, 0, 0, 0],
                   [0, 0, 0, x1, y1, 1],
                   [0, 0, 0, x2, y2, 1],
                   [0, 0, 0, x3, y3, 1]])
    a2 = np.array([u1, u2, u3, v1, v2, v3])
    a = np.linalg.inv(a1) @ a2
    res = np.reshape(np.append(a, [0, 0, 1]), newshape=(3, 3))
    return res


def calculate_model(samples):
    A = create_array(samples)
    return A


def model_error(model, pair):
    return distance.cdist(model @ np.array([pair[0].coords[0], pair[0].coords[1], 1]),
                          np.array([pair[1].coords[0], pair[1].coords[1], 1]), metric='euclidean')


def ransac(pairs, iter, n, max_error):
    bestmodel = None
    bestscore = 0
    for i in range(iter):
        model = None
        indices = np.random.choice(pairs.shape[0], size=n)
        choosen = pairs[indices]
        model = calculate_model(choosen)
        score = 0
        for pair in pairs:
            error = model_error(model, pair)
            if (error < max_error):
                score += 1
        if score > bestscore:
            bestscore = score
            bestmodel = model
    return bestmodel
