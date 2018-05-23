import numpy as np
from scipy.spatial import distance

from utils import loader
from utils import printer

IMAGES_1 = ['data/1/DSC03230.png', 'data/1/DSC03240.png']
IMAGES_2 = ['data/2/DSC_5824.png', 'data/2/DSC_5825.png']

SIFT_SUFFIX = '.haraff.sift'

THRESHOLD = 150
CURRENT = IMAGES_2


def calc_distances(l1, l2):
    v1 = np.array([x.vector for x in l1])
    v2 = np.array([x.vector for x in l2])
    distances = distance.cdist(v1, v2)
    pairs = []
    for i in range(len(l1)):
        corresponding = np.argmin(distances[i, :])
        if (i == np.argmin(distances[:, corresponding]) and distances[i, corresponding] < THRESHOLD):
            pairs.append((l1[i], l2[corresponding]))
    return pairs


def load_sifts(images):
    sift_paths = list(map(lambda x: x + SIFT_SUFFIX, images))
    return tuple(map(loader.load_file, sift_paths))


def main():
    s1, s2 = load_sifts(CURRENT)
    pairs = calc_distances(s1, s2)
    printer.print_image(CURRENT, pairs)


if __name__ == "__main__":
    main()
