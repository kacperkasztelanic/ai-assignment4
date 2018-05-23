import adjacency
from utils import loader
from utils import printer

IMAGES_1 = ['data/1/DSC03230.png', 'data/1/DSC03240.png']
IMAGES_2 = ['data/2/DSC_5824.png', 'data/2/DSC_5825.png']
IMAGES_3 = ['data/3/3-1.png', 'data/3/3-2.png']


CURRENT = IMAGES_2


def main():
    s1, s2 = loader.load_sifts(CURRENT)
    pairs = adjacency.find_pairs_euclidean(s1, s2)
    # adjacency.filter_points(pairs, 5, 0.79)
    # printer.print_image(CURRENT, pairs)
    filtered = adjacency.filter_pairs(pairs, 5, 0.59)
    printer.print_image(CURRENT, filtered)

if __name__ == "__main__":
    main()
