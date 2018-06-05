from enum import Enum
from PIL import Image
import numpy as np
from scipy.spatial import distance

from Point import Point


class DistanceHeuristic(Enum):
    default_none = 0
    distance_heuristic = 1


class Ransac:

    def __init__(self, paths, filtered_pairs, heuristic=DistanceHeuristic.default_none, lower_limit=0.01,
                 upper_limit=0.5):
        self.heuristic = heuristic
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.filtered_pairs = filtered_pairs
        self.init_limits(paths)
        # self.matrix_A = None
        # self.matrix_H = None
        self.model = None
        self._ransac_pairs = []

    def init_limits(self, paths):
        img_1 = Image.open(paths[0])
        img_2 = Image.open(paths[1])
        size = max(sum(img_1.size), sum(img_2.size)) / 2
        self.lower_limit *= size
        self.upper_limit *= size

    def calculate(self, size, no_draws, max_error):
        self.ransac_model(size, no_draws, max_error)
        self.calculate_ransac_pairs()

    def ransac_model(self, size, no_draws, max_error):
        pairs = self.filtered_pairs
        best_model = None
        best_score = 0
        for i in range(no_draws):
            model = None
            while model is None:
                indices = np.random.choice(pairs.shape[0], size=size)
                chosen = pairs[indices]
                if self.heuristic == DistanceHeuristic.default_none:
                    model = self.calc_model(chosen)
                elif self.heuristic == DistanceHeuristic.distance_heuristic:
                    v = np.array([p[0].coords for p in chosen])
                    dist = distance.cdist(v, v, metric='euclidean')
                    np.fill_diagonal(dist, self.upper_limit)
                    if np.min(dist) < self.lower_limit or np.min(dist) > self.upper_limit:
                        model = None
                    else:
                        model = self.calc_model(chosen)
            score = 0
            for pair in pairs:
                error = self.model_error(model, pair)
                if error < max_error:
                    score += 1
            if score > best_score:
                best_score = score
                best_model = model
        self.model = best_model

    def calc_model(self, samples):
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
            a = self.affine_array(x1, y1, x2, y2, x3, y3, u1, v1, u2, v2, u3, v3)
        else:
            x4 = samples[3][0].coords[0]
            y4 = samples[3][0].coords[1]
            u4 = samples[3][1].coords[0]
            v4 = samples[3][1].coords[1]
            a = self.perspective_array(x1, y1, x2, y2, x3, y3, x4, y4, u1, v1, u2, v2, u3, v3, u4, v4)
        return a

    def affine_array(self, x1, y1, x2, y2, x3, y3, u1, v1, u2, v2, u3, v3):
        a1 = np.array([[x1, y1, 1, 0, 0, 0],
                       [x2, y2, 1, 0, 0, 0],
                       [x3, y3, 1, 0, 0, 0],
                       [0, 0, 0, x1, y1, 1],
                       [0, 0, 0, x2, y2, 1],
                       [0, 0, 0, x3, y3, 1]])
        a2 = np.array([u1, u2, u3, v1, v2, v3])
        if not self.is_invertible(a1):
            return None
        a = np.linalg.inv(a1) @ a2
        res = np.reshape(np.append(a, [0, 0, 1]), newshape=(3, 3))
        return res

    @staticmethod
    def is_invertible(a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    def perspective_array(self, x1, y1, x2, y2, x3, y3, x4, y4, u1, v1, u2, v2, u3, v3, u4, v4):
        a1 = np.array([[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1],
                       [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2],
                       [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3],
                       [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4],
                       [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1],
                       [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2],
                       [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3],
                       [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4]])
        a2 = np.array([u1, u2, u3, u4, v1, v2, v3, v4])
        if not self.is_invertible(a1):
            return None
        a = np.linalg.inv(a1) @ a2
        res = np.reshape(np.append(a, 1), newshape=(3, 3))
        return res

    @staticmethod
    def model_error(model, pair):
        return distance.cdist(np.reshape(model @ np.array([pair[0].coords[0], pair[0].coords[1], 1]), newshape=(-1, 1)),
                              np.array([pair[1].coords[0], pair[1].coords[1], 1]).reshape(-1, 1),
                              metric='euclidean').flatten()[0]

    def calculate_ransac_pairs(self):
        self._ransac_pairs = []
        for pair in self.filtered_pairs:
            temp = self.model @ np.array([pair[0].coords[0], pair[0].coords[1], 1])
            self._ransac_pairs.append((pair[0], Point((temp[0], temp[1]))))

    def get_ransac_pairs(self):
        return self._ransac_pairs