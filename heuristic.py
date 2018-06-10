import numpy as np
from PIL import Image
from scipy.spatial import distance


class EuclideanDistanceHeuristic:
    def __init__(self, img_paths, lower_limit=0.01, upper_limit=0.3):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.init_limits(img_paths)

    def init_limits(self, img_paths):
        img_1 = Image.open(img_paths[0])
        img_2 = Image.open(img_paths[1])
        size = max(sum(img_1.size), sum(img_2.size)) / 2
        self.lower_limit *= size
        self.upper_limit *= size

    def are_pairs_correct(self, chosen_pairs):
        v = np.array([p[0].coords for p in chosen_pairs])
        dist = distance.cdist(v, v, metric='euclidean')
        np.fill_diagonal(dist, self.upper_limit)
        min_ = np.min(dist)
        return self.lower_limit < min_ < self.upper_limit