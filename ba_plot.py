import logging
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Evaluation.EvaluationFunctions import _find_path_by_config_with_learner_object, rename_folders, __write_summary
from Evaluation._config import STANDARD_CONFIG_WITH_CO2, AMOUNT_OF_STRINGS, AMOUNT_OF_CLASSIFIERS, CLASSIFIERS, \
    STREAM_STRINGS, STABLE_STRING_IDX, UNSTABLE_STRING_IDX, RENAME_VALUE, \
    AMOUNT_HYPERPARAMETER_TESTS, TOTAL_AMOUNT_HYPERPARAMETERS, AMOUNT_HYPERPARAMETERS_BEFORE, HYPERPARAMETER_KEYS, \
    HYPERPARAMETERS, PROJECT_FOLDER_PATH, SINGLE_CLASSIFIER_FEATURES_TO_TEST, PAIRWISE_CLASSIFIER_FEATURES_TO_TEST, \
    LEARNER_CONFIG_TO_NAMES, HYPERPARAMETERS_NAMES, STREAM_NAMES, UNSTABLE_CONFIG_WITH_CO2, STABLE_CONFIG_WITH_CO2, \
    STANDARD_RUN_ID, RESULTS_DIR_PATH, HYPERPARAMETER_FILE_NAMES
from Evaluation.config_handling import config_to_learner, adl_run_data_from_config

PLOT_DIR_BA = Path('/home/david/bachlorthesis/overleaf/images/plots')
# PLOT_DIR_BA = PROJECT_FOLDER_PATH / 'plots'
COLOR_PALATE = 'colorblind'
SHOW_PLOTS = False
MARKER_SIZE = 10


def plot_standard_on_all_streams() -> None:
    'plot comparision of a set of hyperparameters on nine different problems'
    sns.set_color_codes(COLOR_PALATE)

    # get dfs:
    paths = [_find_path_by_config_with_learner_object(run_id=STANDARD_RUN_ID, config=STANDARD_CONFIG_WITH_CO2, stream_name=stream_name)
             for stream_name in STREAM_STRINGS]
    data_frames = _load_paths(paths)
    logger = logging.getLogger('standart_set_on_all_streams')
    minimum_size = min(map(len, data_frames))
    logger.info(f'comparison of standard set on all streams done on {minimum_size} instances')
    window_size = max(1, minimum_size // 1000)
    logger.info(f'window size chosen: {window_size} instances')
    data = pd.concat(
        [
            (
                df
                .assign(nr_of_active_layers=df['active_layers'].apply(len))
                .loc[:, ['accuracy', 'nr_of_active_layers', 'emissions']]
                .rolling(window_size, center=True ,step=window_size)
                .mean()
                .head(minimum_size // window_size + 1)
                .reset_index(drop=True)
                .assign(instance=lambda x: x.index * window_size)
                .assign(stream_name=STREAM_NAMES[stream_name])
                .set_index(['instance', 'stream_name'])
             )
            for df, stream_name in zip(data_frames, STREAM_STRINGS)
        ]
    ).dropna().sort_index()
    logger.info(f'collecting dfs done')
    sns.set_color_codes(COLOR_PALATE)
    fig, axes = plt.subplots(ncols=3, figsize=(12, 6), layout="constrained")
    g = sns.lineplot(data=data, x='instance', y='accuracy', hue='stream_name', ax=axes[0], errorbar=None)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('Accuracy [%]')

    sns.set_color_codes(COLOR_PALATE)
    sns.scatterplot(data=data, x='instance', y='nr_of_active_layers', markers='o', hue='stream_name', ax=axes[1], s=MARKER_SIZE, legend=False)
    axes[1].set_title('Amount of Active Layers')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('Amount of Active Layers')

    sns.set_color_codes(COLOR_PALATE)
    sns.lineplot(data=data, x='instance', y='emissions', hue='stream_name', ax=axes[2], errorbar=None, legend=False)
    axes[2].set_title('Emissions')
    axes[2].set_xlabel('Number of Instances')
    axes[2].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

    handles, labels = axes[0].get_legend_handles_labels()
    g.legend().remove()
    fig.legend(handles, labels, loc ='lower center', ncols=4, bbox_to_anchor=(0.5, -0.3), bbox_transform=axes[1].transAxes)

    path = PLOT_DIR_BA / 'standard_set_on_all_streams'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'plot', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    logger.info(f'plotting standard vs all done')

def plot_hyperparameter_in_iso() -> None:
    'plot comparision of hyperparameters'
    sns.set_color_codes(COLOR_PALATE)
    logger = logging.getLogger('hyperparameter_comparison')
    paths = []
    indices_of_key_value = {(k, v): set() for k in HYPERPARAMETER_KEYS for v in HYPERPARAMETERS[k]}
    indices_of_stream = {stream_name: set() for stream_name in STREAM_STRINGS}
    for i in range(AMOUNT_HYPERPARAMETER_TESTS):
        config = STANDARD_CONFIG_WITH_CO2.copy()
        # create the config of hyperparameter test:
        key_idx_plus_idx = i % TOTAL_AMOUNT_HYPERPARAMETERS
        tmp = [amount_before > key_idx_plus_idx for amount_before in AMOUNT_HYPERPARAMETERS_BEFORE]
        hyperparameter_key_idx = tmp.index(True) - 1 if True in tmp else len(AMOUNT_HYPERPARAMETERS_BEFORE) - 1
        hyperparameter_idx = key_idx_plus_idx - AMOUNT_HYPERPARAMETERS_BEFORE[hyperparameter_key_idx]
        stream_name = STREAM_STRINGS[i // TOTAL_AMOUNT_HYPERPARAMETERS]
        hyperparameter_key = HYPERPARAMETER_KEYS[hyperparameter_key_idx]
        if hyperparameter_key == 'grace':
            config['grace_type'] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx][0]
            config['grace_period'] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx][1]
        else:
            config[hyperparameter_key] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx]
        # find path by config
        paths.append(_find_path_by_config_with_learner_object(run_id=STANDARD_RUN_ID, config=config, stream_name=stream_name))
        indices_of_key_value[(hyperparameter_key, HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx])].add(i)
        indices_of_stream[stream_name].add(i)
    logger.info('got all the paths')
    minimum_size = get_minimum_length(paths)
    logger.info(f'hyperparameter comparison done on {minimum_size} instances')
    window_size = max(1, minimum_size // 1000)
    logger.info(f'window size chosen: {window_size} instances')

    get_data = lambda path: _get_data_from_path(path, with_active_layers=True, minimum_length=minimum_size, window_size=window_size)

    for hyperparameter_key in HYPERPARAMETER_KEYS:
        df: pd.DataFrame = pd.concat((
            (
                get_data(paths[indices_of_stream[stream_name].intersection(indices_of_key_value[(hyperparameter_key, val)]).pop()])
                .assign(stream_name=stream_name)
                .assign(hyperparameter_value=lambda x: [RENAME_VALUE(val)]*len(x))
            )
            for val in HYPERPARAMETERS[hyperparameter_key]
            for stream_name in STREAM_STRINGS)
        )
        logger.info(f"loading data done for {hyperparameter_key}")

        # compare hyperparameter mean over all stream
        data: pd.DataFrame = (df
                              .drop(columns=['stream_name'])
                              .groupby([df.index, 'hyperparameter_value'])
                              .mean()
                              .reset_index()
                              )

        logger.info(f"calculating mean over streams done for {hyperparameter_key}")

        sns.set_color_codes(COLOR_PALATE)
        fig, axes = plt.subplots(ncols=3, figsize=(12, 6), layout="constrained")
        g = sns.lineplot(data=data, x='instance', y='accuracy', hue='hyperparameter_value', ax=axes[0], errorbar=None)
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Number of Instances')
        axes[0].set_ylabel('Accuracy [%]')
        sns.set_color_codes(COLOR_PALATE)
        sns.scatterplot(data=data, x='instance', y='nr_of_active_layers', markers='o', hue='hyperparameter_value', ax=axes[1], s=MARKER_SIZE, legend=False)
        axes[1].set_title('Amount of Active Layers')
        axes[1].set_xlabel('Number of Instances')
        axes[1].set_ylabel('Amount of Active Layers')
        sns.set_color_codes(COLOR_PALATE)
        sns.lineplot(data=data, x='instance', y='emissions', hue='hyperparameter_value', ax=axes[2], errorbar=None, legend=False)
        axes[2].set_title('Emissions')
        axes[2].set_xlabel('Number of Instances')
        axes[2].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

        handles, labels = axes[0].get_legend_handles_labels()
        g.legend().remove()
        fig.legend(handles, labels, loc ='lower center', ncols=3, bbox_to_anchor=(0.55, -0,3), bbox_transform=axes[1].transAxes)

        path = PLOT_DIR_BA / 'hyperparameter_comparision' / HYPERPARAMETER_FILE_NAMES[hyperparameter_key]
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / 'mean_of_all_streams', bbox_inches='tight')

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()

        logger.info(f'plotting mean done for {hyperparameter_key}')

        # compare hyperparameter per stream
        for stream_name in STREAM_STRINGS:
            # Filter the data for the current stream
            stream_data = df[df['stream_name'] == stream_name]

            # Create subplots: one for each of accuracy, nr_of_active_layers, emissions
            sns.set_color_codes(COLOR_PALATE)
            fig, axes = plt.subplots(ncols=3, figsize=(12, 6), layout="constrained")

            # Plot Accuracy
            sns.set_color_codes(COLOR_PALATE)
            g = sns.lineplot(data=stream_data, x=stream_data.index, y='accuracy', hue='hyperparameter_value', ax=axes[0])
            axes[0].set_title('Accuracy')
            axes[0].set_xlabel('Number of Instances')
            axes[0].set_ylabel('Accuracy [%]')

            # Plot Nr of Active Layers
            sns.set_color_codes(COLOR_PALATE)
            sns.scatterplot(data=stream_data, x=stream_data.index, y='nr_of_active_layers', s=MARKER_SIZE, markers='o', hue='hyperparameter_value', ax=axes[1], legend=False)
            axes[1].set_title('Amount of Active Layers')
            axes[1].set_xlabel('Number of Instances')
            axes[1].set_ylabel('Amount of Active Layers')

            # Plot Emissions
            sns.set_color_codes(COLOR_PALATE)
            sns.lineplot(data=stream_data, x=stream_data.index, y='emissions', hue='hyperparameter_value', ax=axes[2], legend=False)
            axes[2].set_title('Emissions')
            axes[2].set_xlabel('Number of Instances')
            axes[2].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

            handles, labels = axes[0].get_legend_handles_labels()
            g.legend().remove()
            fig.legend(handles, labels, loc ='lower center', ncols=3, bbox_to_anchor=(0.55, -0,3), bbox_transform=axes[1].transAxes)

            title = f"Compare {HYPERPARAMETERS_NAMES[hyperparameter_key]} on Stream: {STREAM_NAMES[stream_name]}"
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.85)
            plt.tight_layout()

            path = PLOT_DIR_BA / 'hyperparameter_comparision' / HYPERPARAMETER_FILE_NAMES[hyperparameter_key]
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(path / stream_name, bbox_inches='tight')

            if SHOW_PLOTS:
                plt.show()
            else:
                plt.close()

            logger.info(f'plotting {stream_name} done for {hyperparameter_key}')


def plot_hyperparameter_stable_vs_unstable() -> None:
    '''compares stable hyperparameter run vs unstable hyperparameter run'''
    sns.set_color_codes(COLOR_PALATE)
    all_configs = [STABLE_CONFIG_WITH_CO2.copy(), UNSTABLE_CONFIG_WITH_CO2.copy()]
    all_stream_names = [STREAM_STRINGS[STABLE_STRING_IDX], STREAM_STRINGS[UNSTABLE_STRING_IDX]]
    paths = [
        _find_path_by_config_with_learner_object(run_id=STANDARD_RUN_ID, config=config, stream_name=stream_name)
        for config, stream_name in zip(all_configs, all_stream_names)
    ]
    data_frames = _load_paths(paths)
    logger = logging.getLogger('stable_vs_unstable')
    logger.info(f'comparison done on {len(data_frames[0])} instances')
    minimum_size = min(map(len, data_frames))
    window_size = max(1, minimum_size // 1000)
    data = pd.concat([
        (
            df
            .assign(nr_of_active_layers=df['active_layers'].apply(len))
            .loc[:, ['accuracy', 'nr_of_active_layers', 'emissions']]
            .rolling(window_size, center=True, step=window_size)
            .mean()
            .head(minimum_size // window_size + 1)
            .reset_index(drop=True)
            .assign(instance=lambda x: np.arange(len(x)) * window_size)
            .assign(run_name='Stable' if i==0 else 'Unstable')
            .set_index(['instance', 'run_name'])
        )
        for i, df in enumerate(data_frames)
    ]).dropna()

    logger.info(f'window size chosen: {window_size} instances')
    logger.info('data collecting done')
    sns.set_color_codes(COLOR_PALATE)
    fig, axes = plt.subplots(ncols=3, figsize=(12, 6), layout="constrained")
    g = sns.lineplot(data=data, x='instance', y='accuracy', hue='run_name', ax=axes[0], errorbar=None)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('Accuracy [%]')
    sns.set_color_codes(COLOR_PALATE)
    sns.scatterplot(data=data, x='instance', y='nr_of_active_layers', markers='o', hue='run_name', ax=axes[1], s=MARKER_SIZE, legend=False)
    axes[1].set_title('Amount of Active Layers')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('Amount of Active Layers')

    sns.set_color_codes(COLOR_PALATE)
    sns.lineplot(data=data, x='instance', y='emissions', hue='run_name', ax=axes[2], errorbar=None, legend=False)
    axes[2].set_title('Emissions')
    axes[2].set_xlabel('Number of Instances')
    axes[2].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

    handles, labels = axes[0].get_legend_handles_labels()
    g.legend().remove()
    fig.legend(handles, labels, loc ='lower center', ncols=2, bbox_to_anchor=(0.5, -0.2), bbox_transform=axes[1].transAxes)

    path = PLOT_DIR_BA / 'stable_vs_unstable'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'stable_vs_unstable', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    logger.info("plotting stable vs unstable done")


def plot_feature_comparision_different() -> None:
    """plots the comparison of the adl model features"""
    logger = logging.getLogger('plot_feature_comparision_different')
    features_to_plot = set(SINGLE_CLASSIFIER_FEATURES_TO_TEST + [feature for pair in PAIRWISE_CLASSIFIER_FEATURES_TO_TEST for feature in pair])
    pairs_per_feature = {feature: {} for feature in features_to_plot}
    for idx in range(AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS):
        current_classifier = CLASSIFIERS[idx % AMOUNT_OF_CLASSIFIERS]
        set_current = set(current_classifier)
        for other_classifier in CLASSIFIERS:
            set_other = set(other_classifier)
            if len(other_classifier) == len(current_classifier) - 1 and len(set_current.difference(set_other)) == 1:
                current_feature = set_current.difference(set_other).pop()
                if other_classifier in pairs_per_feature[current_feature]:
                    raise ValueError("this should never happen")
                else:
                    pairs_per_feature[current_feature][current_classifier] = other_classifier
    logger.info('all index calc done')
    paths = []
    indices_of_classifier = {}
    indices_of_streams = {}
    for i in range(AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS):
        config = STANDARD_CONFIG_WITH_CO2.copy()
        current_classifier = CLASSIFIERS[i % AMOUNT_OF_CLASSIFIERS]
        config['learner'] = config_to_learner(
            *current_classifier,
            grace_period=(config['grace_period'], config['grace_type']),
            with_co2=True
        )
        stream_name = STREAM_STRINGS[i // AMOUNT_OF_CLASSIFIERS]
        paths.append(_find_path_by_config_with_learner_object(run_id=STANDARD_RUN_ID, config=config, stream_name=stream_name))
        indices_of_classifier.setdefault(current_classifier, set()).add(i)
        indices_of_streams.setdefault(stream_name, set()).add(i)

    logger.info('all paths collected done')

    minimum_length = get_minimum_length(paths)
    logging.getLogger('feature_comparison').info(f'feature comparison done on {minimum_length} instances')
    window_size = max(1, minimum_length // 1000)
    logging.getLogger('feature_comparison').info(f'window size chosen: {window_size} instances')
    get_data = lambda path: _get_data_from_path(path, window_size=window_size, with_active_layers=False, minimum_length=minimum_length)

    for feature in features_to_plot:
        mean_of_feature_on_stream_by_combined: Dict[str, pd.DataFrame] = {}
        for stream_name in STREAM_STRINGS:
            combined_per_stream_and_feature = []
            for classifier_with, classifier_without in pairs_per_feature[feature].items():
                idx_classifier_with = indices_of_classifier[classifier_with].intersection(indices_of_streams[stream_name]).pop()
                idx_classifier_without = indices_of_classifier[classifier_without].intersection(indices_of_streams[stream_name]).pop()
                df_classifier_with = get_data(paths[idx_classifier_with]).assign(feature=f'with {feature}').reset_index().set_index(['instance', 'feature'])
                df_classifier_without = get_data(paths[idx_classifier_without]).assign(feature=f'without {feature}').reset_index().set_index(['instance', 'feature'])
                combined = pd.concat([df_classifier_with, df_classifier_without]).assign(accuracy_per_emission=lambda df: df['accuracy'] / df['emissions'])
                combined_per_stream_and_feature.append(combined)
            logger.info(f"collecting all combined dataframes done for {stream_name}")
            mean_of_feature_on_stream_by_combined[stream_name] = reduce(lambda a, b: a + b, combined_per_stream_and_feature) / len(combined_per_stream_and_feature)
            logger.info(f"calculating mean of combined for stream {stream_name} done")

        mean_of_feature = reduce(lambda a, b: a + b, mean_of_feature_on_stream_by_combined.values()) / len(mean_of_feature_on_stream_by_combined)
        logger.info(f"calculating mean over all streams for {feature} done")

        sns.set_color_codes(COLOR_PALATE)
        fig, axes = plt.subplots(ncols=3, figsize=(12, 6), layout="constrained")
        sns.lineplot(data=mean_of_feature, x='instance', y='accuracy', hue='feature', ax=axes[0], errorbar=None, legend=False)

        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Number of Instances')
        axes[0].set_ylabel('Accuracy [%]')

        # Plot Emissions
        sns.set_color_codes(COLOR_PALATE)
        g = sns.lineplot(data=mean_of_feature, x='instance', y='emissions', hue='feature', ax=axes[1], errorbar=None)
        axes[1].set_title('Emissions')
        axes[1].set_xlabel('Number of Instances')
        axes[1].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

        sns.set_color_codes(COLOR_PALATE)
        sns.lineplot(data=mean_of_feature, x='instance', y='accuracy_per_emission', ax=axes[2], hue='feature', errorbar=None, legend=False)
        axes[2].set_title('$\\frac{\\text{Accuracy}}{\\text{Emissions}}$ vs Instances')
        axes[2].set_xlabel('Number of Instances')
        axes[2].set_ylabel('$\\left(\\frac{\\text{Accuracy}}{\\text{Emissions}}\\right)$')

        handles, labels = axes[1].get_legend_handles_labels()
        g.legend().remove()
        fig.legend(handles, labels, loc ='lower center', ncols=2, bbox_to_anchor=(0.5, -0.2), bbox_transform=axes[1].transAxes)

        path = PLOT_DIR_BA / 'feature_comparison' / 'three_plots' / LEARNER_CONFIG_TO_NAMES[feature]
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / 'with_vs_without', bbox_inches='tight')

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()

        logger.info(f'plotting done for {feature}')



def plot_feature_comparision() -> None:
    """plots the comparison of the adl model features"""
    logger = logging.getLogger('feature_comparison')
    features_to_plot = set(SINGLE_CLASSIFIER_FEATURES_TO_TEST + [feature for pair in PAIRWISE_CLASSIFIER_FEATURES_TO_TEST for feature in pair])
    pairs_per_feature = {feature: {} for feature in features_to_plot}
    for idx in range(AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS):
        current_classifier = CLASSIFIERS[idx % AMOUNT_OF_CLASSIFIERS]
        set_current = set(current_classifier)
        for other_classifier in CLASSIFIERS:
            set_other = set(other_classifier)
            if len(other_classifier) == len(current_classifier) - 1 and len(set_current.difference(set_other)) == 1:
                current_feature = set_current.difference(set_other).pop()
                if other_classifier in pairs_per_feature[current_feature]:
                    raise ValueError("this should never happen")
                else:
                    pairs_per_feature[current_feature][current_classifier] = other_classifier
    logger.info('all index calc done')
    paths = []
    indices_of_classifier = {}
    indices_of_streams = {}
    for i in range(AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS):
        config = STANDARD_CONFIG_WITH_CO2.copy()
        current_classifier = CLASSIFIERS[i % AMOUNT_OF_CLASSIFIERS]
        config['learner'] = config_to_learner(
            *current_classifier,
            grace_period=(config['grace_period'], config['grace_type']),
            with_co2=True
        )
        stream_name = STREAM_STRINGS[i // AMOUNT_OF_CLASSIFIERS]
        paths.append(_find_path_by_config_with_learner_object(run_id=STANDARD_RUN_ID, config=config, stream_name=stream_name))
        indices_of_classifier.setdefault(current_classifier, set()).add(i)
        indices_of_streams.setdefault(stream_name, set()).add(i)

    logger.info('all paths collected done')

    minimum_length = get_minimum_length(paths)
    logging.getLogger('feature_comparison').info(f'feature comparison done on {minimum_length} instances')
    window_size = max(1, minimum_length // 1000)
    logging.getLogger('feature_comparison').info(f'window size chosen: {window_size} instances')
    get_data = lambda path: _get_data_from_path(path, window_size=window_size, with_active_layers=False, minimum_length=minimum_length)

    for feature in features_to_plot:
        differences_per_stream: Dict[str, pd.DataFrame] = {}
        for stream_name in STREAM_STRINGS:
            differences_per_stream_and_feature = []
            for classifier_with, classifier_without in pairs_per_feature[feature].items():
                idx_classifier_with = indices_of_classifier[classifier_with].intersection(indices_of_streams[stream_name]).pop()
                idx_classifier_without = indices_of_classifier[classifier_without].intersection(indices_of_streams[stream_name]).pop()
                df_classifier_with = get_data(paths[idx_classifier_with])
                df_classifier_without = get_data(paths[idx_classifier_without])
                diff_on_stream: pd.DataFrame = df_classifier_with - df_classifier_without
                differences_per_stream_and_feature.append(diff_on_stream)
            logger.info(f"collecting all delta dataframes done for {stream_name}")
            differences_per_stream[stream_name] = reduce(lambda a, b: a + b, differences_per_stream_and_feature) / len(differences_per_stream_and_feature)
            logger.info(f"calculating mean of differences for stream {stream_name} done")

        logger.info("start plotting")
        logger.info('collecting all dataframes done for streams')
        three_feature_plots_per_stream(differences_per_stream, feature)
        two_feature_plots_per_stream(differences_per_stream, feature)
        logger.info('plotting per stream done')
        mean_difference: pd.DataFrame = reduce(lambda a, b: a + b, differences_per_stream.values()) / len(differences_per_stream)
        logger.info('calculating mean done')
        three_feature_plots_mean(mean_difference, feature)
        two_feature_plots(mean_difference, feature)
        logger.info(f'plotting feature comparision done {datetime.now()}')

def get_minimum_length(paths):
    path_to_summary = (RESULTS_DIR_PATH / f"runID={STANDARD_RUN_ID}" / 'summary.csv').absolute().as_posix()
    # mogon_project_folder_path = Path('/gpfs/fs1/home/djacoby/ADL/results/runs/runID=58')
    path_string_set = list(path.relative_to(RESULTS_DIR_PATH / f"runID={STANDARD_RUN_ID}").as_posix() for path in paths)
    summary = (pd
               .read_csv(filepath_or_buffer=path_to_summary, sep='\t', usecols=['path', 'amount of instances'])
               .assign(is_in_paths=lambda df: df['path'].apply(lambda x: Path(x).relative_to(RESULTS_DIR_PATH / f"runID={STANDARD_RUN_ID}").as_posix()).isin(path_string_set))
               )
    return int(summary.loc[summary['is_in_paths'], 'amount of instances'].min())


def _load_paths(paths: List[Path]) -> List[pd.DataFrame]:
    data_frames = [_get_data_from_path_naive(path) for path in paths]
    # least_amount_of_rows = min(map(len, data_frames))
    return [df.reset_index() for df in data_frames]


def _get_data_from_path(path: Path, with_active_layers: bool, window_size: int, minimum_length: int) -> pd.DataFrame:
    all_metrics_path = path / "metrics_per_window.pickle"
    logger = logging.getLogger("get_data_from_path")
    if not all_metrics_path.exists():
        logger.error(f"No metrics per window under {path}")
        raise ValueError(f'No metrics file exists under this path: {path}')

    if with_active_layers:
        results_csv = (pd
                       .read_pickle(all_metrics_path)
                       .filter(['accuracy', 'active_layers'])
                       .assign(nr_of_active_layers=lambda df: df['active_layers'].apply(len))
                       .drop(columns=['active_layers'])
                       )
    else:
        results_csv = (pd
                       .read_pickle(all_metrics_path)
                       .filter(['accuracy'])
                       )

    results_csv = (results_csv
                   .rolling(window_size, min_periods=1, step=window_size, center=True)
                   .mean()
                   .head(minimum_length // window_size + 1)
                   .reset_index(drop=True)
                   .assign(instance=lambda x: x.index * window_size)
                   .set_index('instance')
                   )
    if not (path / "emissions.csv").exists():
        logger.error(f"No emissions per window under {path}")
        raise ValueError(f'No emissions file exists under this path: {path}')
    emissions = (pd
                 .read_csv((path / "emissions.csv"), usecols=['emissions'])
                 .head(minimum_length)
                 .rolling(window_size, min_periods=1, step=window_size, center=True)
                 .mean()
                 .head(minimum_length // window_size + 1)
                 .reset_index(drop=True)
                 .assign(instance=lambda x: x.index * window_size)
                 .set_index('instance')
                 )

    return pd.merge(results_csv, emissions, right_index=True, left_index=True).sort_index()

def _get_data_from_path_naive(path: Path) -> pd.DataFrame:
    all_metrics_path = path / "metrics_per_window.pickle"
    logger = logging.getLogger("get_data_from_path")
    if not all_metrics_path.exists():
        logger.error(f"No metrics per window under {path}")
        raise ValueError(f'No metrics file exists under this path: {path}')
    results_csv = (pd.read_pickle(all_metrics_path))
    if not (path / "emissions.csv").exists():
        logger.error(f"No emissions per window under {path}")
        raise ValueError(f'No emissions file exists under this path: {path}')
    results_csv = results_csv.merge(pd.read_csv((path / "emissions.csv"), usecols=['emissions']), right_index=True, left_index=True)
    return results_csv


def two_feature_plots_per_stream(df_per_stream: Dict[str, pd.DataFrame], feature: str) -> None:
    """Plot 2 metrics per stream for given feature and window_size"""
    sns.set_color_codes(COLOR_PALATE)
    # figure
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6), layout="constrained")
    handles, labels = [], []

    for stream_name, df in df_per_stream.items():
        line = sns.lineplot(data=df, x=df.index, y='accuracy', ax=axes[0], label=stream_name, legend=False)
        handles.append(line.lines[0])
        labels.append(stream_name)

        sns.lineplot(data=df, x=df.index, y='emissions', ax=axes[1], label=stream_name, legend=False)

    axes[0].set_title('$\\Delta$ Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    axes[1].set_title('$\\Delta$ Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    # plt.suptitle(title)
    # plt.figlegend(handles, labels, loc = 'lower center', ncol=3, labelspacing=0.)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [list(set(sum(lol, []))) for lol in zip(*lines_labels)]
    lgd = fig.legend(lines, labels, loc ='lower center', ncols=4, bbox_to_anchor=(0, -0.3), bbox_transform=axes[1].transAxes)
    # plt.tight_layout()
    # fig.subplots_adjust(bottom=0.2)
    path = PLOT_DIR_BA / 'feature_comparison' / 'two_plots' / title
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'one_line_per_stream', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def three_feature_plots_per_stream(df_per_stream: Dict[str, pd.DataFrame], feature: str) -> None:
    """Plot 3 metrics per stream for given feature and window_size"""
    sns.set_color_codes(COLOR_PALATE)
    # figure
    fig, axes = plt.subplots(ncols=3, figsize=(12, 6), layout="constrained")
    handles, labels = [], []

    for stream_name, df in df_per_stream.items():
        line = sns.lineplot(data=df, x=df.index, y='accuracy', ax=axes[0], label=stream_name, legend=False)
        handles.append(line.lines[0])
        labels.append(stream_name)

        sns.lineplot(data=df, x=df.index, y='emissions', ax=axes[1], label=stream_name, legend=False)

        df['accuracy_per_emission'] = df['accuracy'] / df['emissions']
        sns.lineplot(data=df, x=df.index, y='accuracy_per_emission', ax=axes[2], label=stream_name, legend=False)

    axes[0].set_title('$\\Delta$ Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    axes[1].set_title('$\\Delta$ Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    axes[2].set_title('$\\frac{\\text{Accuracy}}{\\text{Emissions}}$ vs Instances')
    axes[2].set_xlabel('Number of Instances')
    axes[2].set_ylabel('$\\Delta\\left(\\frac{\\text{Accuracy}}{\\text{Emissions}}\\right)$')

    # layout options
    
    title = LEARNER_CONFIG_TO_NAMES[feature]
    # plt.suptitle(title)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [list(set(sum(lol, []))) for lol in zip(*lines_labels)]
    lgd = fig.legend(lines, labels, loc ='lower center', ncols=4, bbox_to_anchor=(0.55, -0.3), bbox_transform=axes[1].transAxes)

    path = PLOT_DIR_BA / 'feature_comparison' / 'three_plots' / title
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'one_line_per_stream', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def two_feature_plots(x: pd.DataFrame, feature: str) -> None:
    '''plot the mean of a feature in 3 ways in a single figure'''
    sns.set_color_codes(COLOR_PALATE)

    # figure
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

    #  3 subplots
    sns.lineplot(x['accuracy'], ax=axes[0])
    axes[0].set_title('$\\Delta$ Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    sns.lineplot(x['emissions'], ax=axes[1])
    axes[1].set_title('$\\Delta$ Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    # plt.suptitle(title)
    plt.tight_layout()

    path = PLOT_DIR_BA / 'feature_comparison' / 'two_plots' / title
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'mean_of_all_streams', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def three_feature_plots_mean(x: pd.DataFrame, feature: str) -> None:
    '''plot the mean of a feature in 3 ways in a single figure'''
    sns.set_color_codes(COLOR_PALATE)
    # data preparation:

    # figure
    fig, axes = plt.subplots(ncols=3, figsize=(12, 6))

    #  3 subplots
    sns.lineplot(x['accuracy'], ax=axes[0])
    axes[0].set_title('$\\Delta$ Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    sns.lineplot(x['emissions'], ax=axes[1])
    axes[1].set_title('$\\Delta$ Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    x['accuracy_per_emission'] = x['accuracy'] / (x['emissions'])
    sns.lineplot(data=x, x=x.index, y='accuracy_per_emission', ax=axes[2])
    axes[2].set_title('$\\frac{\\Delta\\text{Accuracy}}{\\Delta\\text{Emissions}}$ vs Instances')
    axes[2].set_xlabel('Number of Instances')
    axes[2].set_ylabel('$\\Delta\\left(\\frac{\\text{Accuracy}}{\\text{Emissions}}\\right)$')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    # plt.suptitle(title)
    plt.tight_layout()

    path = PLOT_DIR_BA / 'feature_comparison' / 'three_plots' / title 
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'mean_of_all_streams', bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename=(PROJECT_FOLDER_PATH / 'ba_plot.log').as_posix())
    logger = logging.getLogger("plotting ba")
    logger.info("------------------------------------------")
    logger.info(f"getting started: {datetime.now()}")
    rename_folders(STANDARD_RUN_ID)
    logger.info(f"starting feature comparison: {datetime.now()}")
    plot_feature_comparision()
    logger.info(f"starting different feature comparison: {datetime.now()}")
    plot_feature_comparision_different()
    logger.info(f"starting hyperparameter comparison: {datetime.now()}")
    plot_hyperparameter_in_iso()
    logger.info(f"starting stable vs unstable comparison: {datetime.now()}")
    plot_hyperparameter_stable_vs_unstable()
    logger.info(f"starting standard vs all comparison: {datetime.now()}")
    plot_standard_on_all_streams()
    logger.info(f"plotting ba ended: {datetime.now()}")
    logger.info("------------------------------------------")
