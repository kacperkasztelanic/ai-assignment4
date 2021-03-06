from timeit import default_timer as timer

import adjacency
import heuristic
import ransac
from utils import loader
from utils import printer
from utils.timing import timing

IMAGES = {1: ['data/1/DSC03230.png', 'data/1/DSC03240.png'],
          2: ['data/2/DSC_5824.png', 'data/2/DSC_5825.png'],
          3: ['data/3/3-1.png', 'data/3/3-2.png'],
          4: ['data/4/4-1.png', 'data/4/4-2.png'],
          5: ['data/5/5-1.png', 'data/5/5-2.png'],
          6: ['data/7/7-1.png', 'data/7/7-2.png'],
          7: ['data/8/8-1.png', 'data/8/8-2.png'],
          8: ['data/p1/p1-1.png', 'data/p1/p1-2.png'],
          9: ['data/p2/p2-1.png', 'data/p2/p2-2.png'],
          10: ['data/p3/p3-1.png', 'data/p3/p3-2.png'],
          11: ['data/p4/p4-1.png', 'data/p4/p4-2.png'],
          12: ['data/p5/p5-1.png', 'data/p5/p5-2.png']}

PATHS = IMAGES[12]


@timing
def main():
    key_points_1, key_points_2 = loader.load_sifts(PATHS)
    pairs = adjacency.corresponding_points(key_points_1, key_points_2)

    print('allPairs: {0}'.format(len(pairs)))
    printer.print_image(PATHS, pairs, 'allPairs.png')

    n = 25
    t = 0.81
    filtered_pairs = adjacency.filter_pairs(pairs, n=n, threshold=t)
    printer.print_image(PATHS, filtered_pairs, 'adjacency_n{}_t{}.png'.format(n, t))
    print('filteredPairs: {0}'.format(len(filtered_pairs)))
    ransac_ = ransac.Ransac(pairs, filtered_pairs)

    # h = None
    # h = heuristic.EuclideanDistanceHeuristic(PATHS, lower_limit=0.00, upper_limit=0.3)
    h = heuristic.ShapeHeuristic(upper_limit=0.3)
    i = 500
    size = 3
    e = 10
    start = timer()
    ransac_.calculate(size=size, no_draws=i, max_error=e, heuristic=h)
    end = timer()
    print('ransac took: {} s'.format((end - start)))
    ransac_pairs = ransac_.get_ransac_pairs()
    print('ransac pairs: {}'.format(len(ransac_pairs)))

    s = size
    printer.print_image(PATHS, ransac_pairs, 'ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))
    printer.print_all_image(PATHS, key_points_1, key_points_2, filtered_pairs, ransac_pairs,
                            'all_ransac_n{}_t{}_s{}_i{}_e{}.png'.format(n, t, s, i, e))


if __name__ == "__main__":
    main()
