import numpy as np

from Point import Point


def load_file(path):
    res = []
    with open(path, 'r') as data_file:
        data_file.readline()
        n = int(data_file.readline())
        for i in range(n):
            line = data_file.readline().split()
            x = float(line[0])
            y = float(line[1])
            array = np.array(line[5:], dtype=int)
            res.append(Point(x, y, array))
    return res
