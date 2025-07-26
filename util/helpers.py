import os
import re

import pandas as pd
import yaml

from util.constants import EXPERIMENT_CLASSES, DATASET_OUTPUT_PATH


def walk_experiments_directory(root_dir, experiments_dir):
    results = []

    for root, dirs, files in os.walk(os.path.join(root_dir, experiments_dir)):
        for file in files:
            if file.endswith('.yaml'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    experiment_info = yaml.safe_load(f)
                    results.append({'path': f"{experiments_dir}/{os.path.basename(root)}",  # e.g. "Sampling_Experiments/Experiment_1"
                                    # NOTE: we explicitly don't add root, or use os.path.join, as we want a path that is
                                    # (a) relative and (b) platform independent
                                    'experiment_info': experiment_info})

    # ensure that results are ordered by their number. this is not guaranteed by default, as:
    # 1. os.walk makes no guarantees on order
    # 2. the order happens to be alphabetical, thus sorting 10 before 2, etc
    def extract_number(s):
        match = re.search(r'_(\d+)$', s)
        if not match:
            raise ValueError('No experiment number found in path name')
        return int(match.group(1))

    results = sorted(results, key=lambda x: extract_number(x['path']))

    return results

def walk_experiments_directories(root_dir=None):
    results = dict()
    for experiment_class in EXPERIMENT_CLASSES:
        results[experiment_class] = walk_experiments_directory(root_dir=root_dir, experiments_dir=experiment_class)
    return results

def load_experiments(experiments=None, root_dir=None):
    if root_dir is None:
        root_dir = DATASET_OUTPUT_PATH
    if experiments is None:
        experiments = walk_experiments_directories(root_dir=root_dir)

    for exp_type in experiments:
        for experiment in experiments[exp_type]:
            experiment['upstream'] = []
            for i in [1, 2]:
                upstream_sensor_layer_path = os.path.join(root_dir, experiment['path'], "raw", f"upstream-sensor_{i}.csv")
                df_upstream = pd.read_csv(upstream_sensor_layer_path, delimiter=',', parse_dates=['time'])
                df_upstream.set_index('time', inplace=True)
                experiment['upstream'].append(df_upstream)

            experiment['layers'] = []
            for i in range(4):
                sensor_layer_path = os.path.join(root_dir, experiment['path'], "raw", f"sensor-layer_{i}.csv")
                df_layer = pd.read_csv(sensor_layer_path, delimiter=',', parse_dates=['time'])
                df_layer.set_index('time', inplace=True)
                experiment['layers'].append(df_layer)

    return experiments

def group_columns_by_top_level(df):
    """
    re-index to bunch by top-level
    The reason this may be required is that insertion into a multi-index always appends at the end, ignoring first-level
    sort order. It may be possible to avoid the need for this by using a CategoricalIndex with ordered=True

    Careful: This is not an in-place operation!
    :param df: DataFrame
    :return: re-indexed DataFrame
    """
    cols = list(df.columns)
    new_order = [col for group in dict.fromkeys(col[0] for col in cols)
                      for col in cols if col[0] == group]
    return df.reindex(columns=new_order)
