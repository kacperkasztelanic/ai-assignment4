import adjacency
import ransac
from utils import loader
from utils import printer
from utils.timing import timing

IMAGES_1 = ['data/1/DSC03230.png', 'data/1/DSC03240.png']
IMAGES_2 = ['data/2/DSC_5824.png', 'data/2/DSC_5825.png']
IMAGES_3 = ['data/3/3-1.png', 'data/3/3-2.png']

CURRENT = IMAGES_1


@timing
def main():
    key_points_1, key_points_2 = loader.load_sifts(CURRENT)
    print(key_points_1)
    pairs = adjacency.find_pairs_euclidean(key_points_1, key_points_2)

    n = 25
    t = 0.81
    filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
    printer.print_image(CURRENT, filtered_pairs, 'adjacency_n{}_t{}.png'.format(n, t))

    i = 500
    s = 3
    e = 1
    model = ransac.ransac_model(filtered_pairs, s=s, iter=i, max_error=e)
    ransac_pairs = ransac.ransac_pairs(filtered_pairs, model)
    printer.print_image(CURRENT, ransac_pairs, 'ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
    printer.print_all_image(CURRENT, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
                            'all_ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))


if __name__ == "__main__":
    main()
