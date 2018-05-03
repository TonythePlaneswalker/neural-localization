from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def ray_cast(map):
    shape = (map.shape[0], map.shape[1], 4)
    Z = np.zeros(shape, dtype=np.uint8)
    for i in range(0, map.shape[0]):
        for j in range(0, map.shape[1]):
            if map[i][j]:
                left = 1
                while map[i][j - left]:
                    left += 1
                Z[i][j][3] = left
                up = 1
                while map[i - up][j]:
                    up += 1
                Z[i][j][0] = up
                right = 1
                while map[i][j + right]:
                    right += 1
                Z[i][j][1] = right
                down = 1
                while map[i + down][j]:
                    down += 1
                Z[i][j][2] = down
    return Z
