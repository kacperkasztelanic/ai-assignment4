import adjacency
import ransac
from utils import csv_writer
from utils import loader
from utils import printer

IMAGES_1 = ['data/1/DSC03230.png', 'data/1/DSC03240.png']
IMAGES_2 = ['data/2/DSC_5824.png', 'data/2/DSC_5825.png']
IMAGES_3 = ['data/3/3-1.png', 'data/3/3-2.png']

CURRENT = IMAGES_1


def main():
    s1, s2 = loader.load_sifts(CURRENT)
    pairs = adjacency.find_pairs_euclidean(s1, s2)
    filtered = adjacency.filter_pairs(pairs, 5, 0.8)
    printer.print_image(CURRENT, filtered)
    csv_writer.save(pairs, 'image1_all.txt')
    csv_writer.save(filtered, 'image1_n5_t08.txt')
    a = ransac.calculate_model([filtered[0], filtered[1], filtered[2]])
    print(a)


if __name__ == "__main__":
    main()
