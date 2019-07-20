import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import sys
import glob
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy import misc
import random
import numpy as np
from torchvision.transforms import ToTensor


def Dist_map(sil_CP , sil_GT):
    shapes = []
    shapes.extend(sil_CP)
    shapes.extend(sil_GT)

    for i in enumerate(shapes):
        edge_map = np.asarray(i)
        edge_val = np.max(edge_map)

        # initialize distance map
        distance_map = np.empty(edge_map.shape)

        # caclulate distance map
        for coordinates, value in np.ndenumerate(distance_map):
            # for each coordinate, calculate its L1 norm from all the edges
            distances_from_edges = [abs(coordinates[0] - ind[0]) + abs(coordinates[1] - ind[1])
                                    for ind, val in np.ndenumerate(edge_map) if val == edge_val]
            # find the minimum of the above values
            distance_map[coordinates] = min(distances_from_edges)

        # # plot the edge maps above their distance maps
        # plt.subplot(2, len(shapes), i + 1)
        # plt.imshow(edge_map, cmap='gray')
        # plt.subplot(2, len(shapes), i + 1 + len(shapes))
        # plt.imshow(distance_map, cmap='gray')

    sil_cp = sil_CP.detach().cpu().numpy().transpose((1, 2, 0))
    sil_cp = np.squeeze((sil_cp * 255)).astype(np.uint8)
    sil_GT = sil_GT.detach().cpu().numpy().transpose((1, 2, 0))
    sil_GT = np.squeeze((sil_GT * 255)).astype(np.uint8)

    a = plt.subplot(1,2, 1)
    a.set_title('ground truth')
    plt.imshow(sil_GT, cmap='gray')

    a = plt.subplot(1, 2, 2)
    a.set_title('computed')
    plt.imshow(sil_cp, cmap='gray')

    plt.show()



    return sil_CP, sil_GT