import logging
from pathlib import Path
from typing import List

import pandas as pd

from Evaluation.EvaluationFunctions import _find_path_by_config_with_learner_object, rename_folders
from Evaluation._config import STANDARD_CONFIG_WITH_CO2, AMOUNT_OF_STRINGS, AMOUNT_OF_CLASSIFIERS, CLASSIFIERS, \
    STREAM_STRINGS, UNSTABLE_CONFIG, STABLE_CONFIG, STABLE_STRING_IDX, UNSTABLE_STRING_IDX, RESULTS_DIR_PATH, \
    AMOUNT_HYPERPARAMETER_TESTS, TOTAL_AMOUNT_HYPERPARAMETERS, AMOUNT_HYPERPARAMETERS_BEFORE, HYPERPARAMETER_KEYS, \
    HYPERPARAMETERS, PROJECT_FOLDER_PATH
from Evaluation.config_handling import config_to_learner


def plot_hyperparameter_in_iso() -> None:
    paths = []
    all_configs = [STANDARD_CONFIG_WITH_CO2.copy() for _ in range(AMOUNT_HYPERPARAMETER_TESTS)]
    for i, config in enumerate(all_configs):
        # create the config of hyperparameter test:
        key_idx_plus_idx = i % TOTAL_AMOUNT_HYPERPARAMETERS
        tmp = [amount_before > key_idx_plus_idx for amount_before in AMOUNT_HYPERPARAMETERS_BEFORE]
        hyperparameter_key_idx = tmp.index(True) - 1 if True in tmp else len(AMOUNT_HYPERPARAMETERS_BEFORE) - 1
        hyperparameter_idx = key_idx_plus_idx - AMOUNT_HYPERPARAMETERS_BEFORE[hyperparameter_key_idx]
        stream_idx = i // TOTAL_AMOUNT_HYPERPARAMETERS
        hyperparameter_key = HYPERPARAMETER_KEYS[hyperparameter_key_idx]
        if hyperparameter_key == 'grace':
            config['grace_type'] = HYPERPARAMETERS[hyperparameter_key][0]
            config['grace_period'] = HYPERPARAMETERS[hyperparameter_key][1]
        else:
            config[hyperparameter_key] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx]

        # find path by config
        paths.append(_find_path_by_config_with_learner_object(run_id=99, config=config, stream_name=STREAM_STRINGS[stream_idx]))

    data_frames = _load_paths(paths)
    for df in data_frames:
        print(df.head(5))
    # todo: plot all
    raise NotImplemented


def plot_hyperparameter_stable_vs_unstable() -> None:
    all_configs = [STABLE_CONFIG.copy(), UNSTABLE_CONFIG.copy()]
    all_stream_names = [STREAM_STRINGS[STABLE_STRING_IDX], STREAM_STRINGS[UNSTABLE_STRING_IDX]]
    paths = [
        _find_path_by_config_with_learner_object(run_id=99, config=config, stream_name=stream_name) 
        for config, stream_name in zip(all_configs, all_stream_names)
    ]
    data_frames = _load_paths(paths)
    for df in data_frames:
        print(df.head(5))
    # todo: plot all
    raise NotImplemented


def plot_feature_comparision() -> None:
    paths = []
    all_configs = [STANDARD_CONFIG_WITH_CO2.copy() for _ in range(AMOUNT_OF_STRINGS) for _  in range(AMOUNT_OF_CLASSIFIERS)]
    for i, config in enumerate(all_configs):
        config['learner'] = config_to_learner(
            *CLASSIFIERS[i % AMOUNT_OF_CLASSIFIERS],
            grace_period=(config['grace_period'], config['grace_type']), 
            with_co2=True
        )
        stream_name = STREAM_STRINGS[i // AMOUNT_OF_CLASSIFIERS]
        paths.append(_find_path_by_config_with_learner_object(run_id=99, config=config, stream_name=stream_name))
    data_frames = _load_paths(paths)
    print(data_frames)
    for df in data_frames:
        print(df.head(5))
    # todo: plot all
    raise NotImplementedError


def _load_paths(paths: List[Path]) -> List[pd.DataFrame]:
    data_frames = [_get_data_from_path(path) for path in paths]
    least_amount_of_rows = min(map(len, data_frames))
    return [df.head(least_amount_of_rows) for df in data_frames]


def _get_data_from_path(path: Path) -> pd.DataFrame:
    all_metrics_path = path / "metrics_per_window.pickle"
    logger = logging.getLogger("get_data_from_path")
    if not all_metrics_path.exists():
        logger.error(f"No metrics per window under {path}")
        raise ValueError(f'No metrics file exists under this path: {path}')
    results_csv = pd.read_pickle(all_metrics_path)
    if not (path / "emissions.csv").exists():
        logger.error(f"No emissions per window under {path}")
        raise ValueError(f'No emissions file exists under this path: {path}')
    results_csv = results_csv.merge(pd.read_csv(path / "emissions.csv"), right_index=True, left_index=True)
    return results_csv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename=(PROJECT_FOLDER_PATH / 'ba_plot.log').as_posix())
    rename_folders(99)
    plot_feature_comparision()
    # plot_hyperparameter_in_iso()
    # plot_hyperparameter_stable_vs_unstable()
