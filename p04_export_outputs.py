import os
import pickle

import pandas as pd
from matplotlib import pyplot as plt

from util.constants import DATASET_OUTPUT_PATH, EXPERIMENT_CLASSES, SENSOR_SPECIMEN, ALL_SENSOR_TYPES, \
    LAYER_SENSOR_SPECIMEN

if __name__ == '__main__':
    # display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100000)

    """ load data """
    with open(os.path.join(DATASET_OUTPUT_PATH, 'experiments.pkl'), "rb") as experiments_file:
        experiments = pickle.load(experiments_file)

    """ export CSVs """
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                csv_path = os.path.join(DATASET_OUTPUT_PATH, experiment['path'], 'preprocessed', f'{key}_{idx}.csv')
                print(f"Exporting {os.path.normpath(csv_path)} ...")
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                with open(csv_path, "wb") as preprocessed_csv:
                    df = experiment[key][idx]
                    df_flat = df.copy()
                    # flatten headers: concatenate levels with dots. strip trailing dots that remain from single-level columns
                    df_flat.columns = ['.'.join(col).strip('.') for col in df.columns.values]
                    df_flat.to_csv(preprocessed_csv)
            print()
    print("Done.")

    """ show some example plots """
    for sensor_type in ALL_SENSOR_TYPES:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for key, idx in SENSOR_SPECIMEN:
            ax1.plot(experiments["Sampling_Experiments"][0][key][idx][sensor_type]['ppm'].rolling(f'30s').mean(),
                     label=f'{key} {idx}')
            ax1.set_ylabel('ppm')
        ax1.grid()
        ax1.legend()
        for key, idx in LAYER_SENSOR_SPECIMEN:
            ax2.plot(experiments["Sampling_Experiments"][0][key][idx][sensor_type]['ppm_relative'].rolling(f'30s').mean(),
                     label=f'{key} {idx}')
            ax2.set_ylabel('ppm above background')
        ax2.grid()
        ax2.legend()

        plt.suptitle(f"{sensor_type} in Sampling Experiment 1 (smoothed)")
        plt.show()
