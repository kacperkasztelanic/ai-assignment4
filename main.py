import numpy as np
from scipy.spatial import distance

from utils import loader


def calc_distances(l1, l2):
    v1 = [x.vector for x in l1]
    v2 = [x.vector for x in l2]
    distances = distance.cdist(v1, v2)
    pairs = []
    for i in range(len(l1)):
        amin = np.argmin(distances[i, :])
        if (amin < len(l2) and i == np.argmin(distances[:, amin])):
            if (distances[i, amin] < 150):
                pairs.append((l1[i], l2[amin]))
    return pairs


def helper():
    list = loader.load_file('data/1/DSC03230.png.haraff.sift')
    list2 = loader.load_file('data/1/DSC03240.png.haraff.sift')
    pairs = calc_distances(list, list2)
    return pairs


def main():
    list = loader.load_file('data/1/DSC03230.png.haraff.sift')
    list2 = loader.load_file('data/1/DSC03240.png.haraff.sift')
    pairs = calc_distances(list, list2)


if __name__ == "__main__":
    main()
