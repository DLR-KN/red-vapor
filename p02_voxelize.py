import os
import pickle
import warnings

import pandas as pd
from pandas.errors import PerformanceWarning

from util.constants import DATASET_OUTPUT_PATH, LAYER_SENSOR_SPECIMEN

if __name__ == '__main__':
    # display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100000)

    """ load data """
    with open(os.path.join(DATASET_OUTPUT_PATH, 'experiments.pkl'), "rb") as experiments_file:
        experiments = pickle.load(experiments_file)


    """ turn the sampling experiments into voxel maps by grouping them by their x, y and z coordinates and averaging """
    voxel_maps = []
    # for exp_class in ["Sampling_Experiments"]:
    exp_class = "Sampling_Experiments"
    for exp_idx, experiment in enumerate(experiments[exp_class]):
        print(f"Voxelizing experiment {experiment['path']}")
        layer_dfs = []
        for key, idx in LAYER_SENSOR_SPECIMEN:
            df = experiment[key][idx]
            layer_dfs.append(df)
        combined_df = pd.concat(layer_dfs)

        with warnings.catch_warnings():
            # ignore performance penalty warning
            warnings.simplefilter("ignore", PerformanceWarning)

            combined_df = combined_df.drop('trig', axis=1)  # averaging over trig is meaningless
            voxelized = combined_df.groupby(['x', 'y', 'z']).mean(numeric_only=True).reset_index()
            voxel_maps.append({
                'path': experiment['path'],
                'experiment_info': experiment['experiment_info'],
                'voxel_map': voxelized,
                })

    """ export """
    # export Python pickle
    print("Exporting...")
    with open(os.path.join(DATASET_OUTPUT_PATH, 'voxel_maps.pkl'), "wb") as voxel_maps_file:
        pickle.dump(voxel_maps, voxel_maps_file)

    print("Done")
