import os
import pickle
import warnings

import meshio
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from util.constants import DATASET_OUTPUT_PATH

CELL_SIZE = [0.12, 0.12, 0.08]
POINT_ORDER = [[-1., -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]


if __name__ == '__main__':
    # display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100000)

    """ load data """
    with open(os.path.join(DATASET_OUTPUT_PATH, 'voxel_maps.pkl'), "rb") as voxel_maps_file:
        voxel_maps = pickle.load(voxel_maps_file)

    for voxel_map in voxel_maps:
        df = voxel_map["voxel_map"]
        cells_center = np.vstack((df['x'].values, df['y'].values, df['z'].values)).T

        with warnings.catch_warnings():
            # ignore performance penalty warning
            warnings.simplefilter("ignore", PerformanceWarning)

            # flatten headers: concatenate levels with dots. strip trailing dots that remain from single-level columns
            df_flat = df.copy()
            df_flat = df_flat.drop(['x', 'y', 'z',
                                    'x_reported', 'y_reported', 'z_reported',
                                    'wind-u', 'wind-v', 'wind-w'], axis=1)
            df_flat.columns = ['.'.join(col).strip('.') for col in df_flat.columns.values]

            # pack it into list-in-list-in-dict, as meshio wants
            # output = df_flat.to_dict('list')  # not enough
            output = {col: [df_flat[col].tolist()] for col in df_flat.columns}

        # combine wind to 3D vector
        output["wind"] = [np.vstack((df['wind-u'].values, df['wind-v'].values, df['wind-w'].values)).T]

        points = []
        cells = []

        for i, c in enumerate(cells_center):
            print(f"Processing {voxel_map["path"]} cell {i} of {len(cells_center)}...")
            p_idxs = []
            for d in POINT_ORDER:
                p = (c[0] + d[0] * CELL_SIZE[0] / 2,
                     c[1] + d[1] * CELL_SIZE[1] / 2,
                     c[2] + d[2] * CELL_SIZE[2] / 2)

                try:
                    p_idx = points.index(p)
                    p_idxs.append(p_idx)
                except ValueError:  # point not in list yet
                    points.append(p)
                    p_idxs.append(len(points) - 1)

            cells.append(p_idxs)

        mesh = meshio.Mesh(points, [("hexahedron", cells), ], cell_data=output, )
        mesh.write(os.path.join(DATASET_OUTPUT_PATH, voxel_map["path"], "voxel_map.vtk"))
