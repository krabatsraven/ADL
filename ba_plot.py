import logging
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Evaluation.EvaluationFunctions import _find_path_by_config_with_learner_object, rename_folders
from Evaluation._config import STANDARD_CONFIG_WITH_CO2, AMOUNT_OF_STRINGS, AMOUNT_OF_CLASSIFIERS, CLASSIFIERS, \
    STREAM_STRINGS, UNSTABLE_CONFIG, STABLE_CONFIG, STABLE_STRING_IDX, UNSTABLE_STRING_IDX, RENAME_VALUE, \
    AMOUNT_HYPERPARAMETER_TESTS, TOTAL_AMOUNT_HYPERPARAMETERS, AMOUNT_HYPERPARAMETERS_BEFORE, HYPERPARAMETER_KEYS, \
    HYPERPARAMETERS, PROJECT_FOLDER_PATH, SINGLE_CLASSIFIER_FEATURES_TO_TEST, PAIRWISE_CLASSIFIER_FEATURES_TO_TEST, \
    LEARNER_CONFIG_TO_NAMES, HYPERPARAMETERS_NAMES, STREAM_NAMES
from Evaluation.config_handling import config_to_learner


PLOT_DIR_BA = Path('/home/david/bachlorthesis/overleaf/images/plots')
COLOR_PALATE = 'colorblind'


def plot_hyperparameter_in_iso() -> None:
    'plot comparision of hyperparameters'
    paths = []
    indices_of_key_value = {(k, v): set() for k in HYPERPARAMETER_KEYS for v in HYPERPARAMETERS[k]}
    indices_of_stream = {stream_name: set() for stream_name in STREAM_STRINGS}
    all_configs = [STANDARD_CONFIG_WITH_CO2.copy() for _ in range(AMOUNT_HYPERPARAMETER_TESTS)]
    for i, config in enumerate(all_configs):
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
        paths.append(_find_path_by_config_with_learner_object(run_id=99, config=config, stream_name=stream_name))
        indices_of_key_value[(hyperparameter_key, HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx])].add(i)
        indices_of_stream[stream_name].add(i)

    data_frames = _load_paths(paths)
    logger = logging.getLogger('hyperparameter_comparison') 
    logger.info(f'hyperparameter comparison done on {len(data_frames[0])} instances')
    window_size = max(1, len(data_frames[0]) // 1000)
    logger.info(f'window size chosen: {window_size} instances')
    data_frames = [
        (
            df
            .assign(nr_of_active_layers=df['active_layers'].apply(len))
            .loc[:, ['accuracy', 'nr_of_active_layers', 'emissions']]
            .rolling(window_size, center=True ,step=window_size)
            .mean()
        )
        for df in data_frames
    ]

    for hyperparameter_key in HYPERPARAMETER_KEYS:
        df: pd.DataFrame = pd.concat((
            (
                data_frames[indices_of_stream[stream_name].intersection(indices_of_key_value[(hyperparameter_key, val)]).pop()]
                .assign(stream_name=stream_name)
                .assign(hyperparameter_value=[RENAME_VALUE(val)]*len(data_frames[0]))
            )
            for val in HYPERPARAMETERS[hyperparameter_key]
            for stream_name in STREAM_STRINGS)
        )

        # compare hyperparameter mean over all stream
        data: pd.DataFrame = (df
                              .drop(columns=['stream_name'])
                              .groupby([df.index, 'hyperparameter_value'])
                              .mean()
                              .reset_index()
                              )

        fig, axes = plt.subplots(ncols=3)
        g = sns.lineplot(data=data, x='level_0', y='accuracy', hue='hyperparameter_value', ax=axes[0], errorbar=None)
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Number of Instances')
        axes[0].set_ylabel('Accuracy [%]')

        sns.lineplot(data=data, x='level_0', y='nr_of_active_layers', hue='hyperparameter_value', ax=axes[1], errorbar=None, legend=False)
        axes[1].set_title('Amount of Active Layers')
        axes[1].set_xlabel('Number of Instances')
        axes[1].set_ylabel('Amount of Active Layers')

        sns.lineplot(data=data, x='level_0', y='emissions', hue='hyperparameter_value', ax=axes[2], errorbar=None, legend=False)
        axes[2].set_title('Emissions')
        axes[2].set_xlabel('Number of Instances')
        axes[2].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

        handles, labels = axes[0].get_legend_handles_labels()
        g.legend().remove()
        fig.legend(handles, labels, loc ='lower center', ncols=3, bbox_to_anchor=(0.55, 0))

        title = f"Compare Different {HYPERPARAMETERS_NAMES[hyperparameter_key]}"
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.85)
        plt.tight_layout()

        path = PLOT_DIR_BA / 'hyperparameter_comparision' / title
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / 'mean_of_all_streams', bbox_inches='tight')

        plt.show()

        # compare hyperparameter per stream
        for stream_name in STREAM_STRINGS:
            # Filter the data for the current stream
            stream_data = df[df['stream_name'] == stream_name]

            # Create subplots: one for each of accuracy, nr_of_active_layers, emissions
            fig, axes = plt.subplots(ncols=3)

            # Plot Accuracy
            g = sns.lineplot(data=stream_data, x=stream_data.index, y='accuracy', hue='hyperparameter_value', ax=axes[0])
            axes[0].set_title('Accuracy')
            axes[0].set_xlabel('Number of Instances')
            axes[0].set_ylabel('Accuracy [%]')

            # Plot Nr of Active Layers
            sns.lineplot(data=stream_data, x=stream_data.index, y='nr_of_active_layers', hue='hyperparameter_value', ax=axes[1], legend=False)
            axes[1].set_title('Amount of Active Layers')
            axes[1].set_xlabel('Number of Instances')
            axes[1].set_ylabel('Amount of Active Layers')

            # Plot Emissions
            sns.lineplot(data=stream_data, x=stream_data.index, y='emissions', hue='hyperparameter_value', ax=axes[2], legend=False)
            axes[2].set_title('Emissions')
            axes[2].set_xlabel('Number of Instances')
            axes[2].set_ylabel('Emissions [$kg\\ CO_2 \\text{equiv}$]')

            handles, labels = axes[0].get_legend_handles_labels()
            g.legend().remove()
            fig.legend(handles, labels, loc ='lower center', ncols=3, bbox_to_anchor=(0.55, 0))
    
            title = f"Compare {HYPERPARAMETERS_NAMES[hyperparameter_key]} on Stream: {STREAM_NAMES[stream_name]}"
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.85)
            plt.tight_layout()
    
            path = PLOT_DIR_BA / 'hyperparameter_comparision' / HYPERPARAMETERS_NAMES[hyperparameter_key]
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(path / stream_name, bbox_inches='tight')

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Make space for the main title
            plt.show()


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
    raise NotImplementedError


def plot_feature_comparision() -> None:
    """plots the comparison of the adl model features"""
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
                    continue
                else:
                    pairs_per_feature[current_feature][current_classifier] = other_classifier

    paths = []
    all_configs = [STANDARD_CONFIG_WITH_CO2.copy() for _ in range(AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS)]
    indices_of_classifier = {}
    indices_of_streams = {}
    for i, config in enumerate(all_configs):
        current_classifier = CLASSIFIERS[i % AMOUNT_OF_CLASSIFIERS]
        config['learner'] = config_to_learner(
            *current_classifier,
            grace_period=(config['grace_period'], config['grace_type']),
            with_co2=True
        )
        stream_name = STREAM_STRINGS[i // AMOUNT_OF_CLASSIFIERS]
        paths.append(_find_path_by_config_with_learner_object(run_id=99, config=config, stream_name=stream_name))
        indices_of_classifier.setdefault(current_classifier, set()).add(i)
        indices_of_streams.setdefault(stream_name, set()).add(i)

    data_frames = _load_paths(paths)
    logging.getLogger('feature_comparison').info(f'feature comparison done on {len(data_frames[0])} instances')
    window_size = max(1, len(data_frames[0]) // 1000)
    logging.getLogger('feature_comparison').info(f'window size chosen: {window_size} instances')
    data_frames = [df.loc[:, ['accuracy', 'emissions']] for df in data_frames]

    for feature in features_to_plot:
        differences_per_stream_and_feature = {stream_name: [] for stream_name in STREAM_STRINGS}
        for lhs, rhs in pairs_per_feature[feature].items():
            for stream_name in STREAM_STRINGS:
                idx_of_lhs_on_stream = indices_of_classifier[lhs].intersection(indices_of_streams[stream_name]).pop()
                idx_of_rhs_on_stream = indices_of_classifier[rhs].intersection(indices_of_streams[stream_name]).pop()
                diff_on_stream = data_frames[idx_of_lhs_on_stream] - data_frames[idx_of_rhs_on_stream]
                differences_per_stream_and_feature[stream_name].append(diff_on_stream)
        differences_per_stream = {stream_name: pd.concat(differences_per_stream_and_feature[stream_name]).groupby(level=0).mean() for stream_name in STREAM_STRINGS}
        mean_difference: pd.DataFrame = pd.concat(differences_per_stream.values()).groupby(level=0).mean()

        three_feature_plots_mean(mean_difference, feature, window_size)
        three_feature_plots_per_stream(differences_per_stream, feature, window_size)
        two_feature_plots(mean_difference, feature, window_size)
        two_feature_plots_per_stream(differences_per_stream, feature, window_size)


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


def two_feature_plots_per_stream(df_per_stream: Dict[str, pd.DataFrame], feature: str, window_size: int) -> None:
    """Plot 2 metrics per stream for given feature and window_size"""
    sns.set_color_codes(COLOR_PALATE)
    # figure
    fig, axes = plt.subplots(ncols=2)
    handles, labels = [], []

    for stream_name, df in df_per_stream.items():
        df = df.rolling(window_size, center=True ,step=window_size).mean()
        line = sns.lineplot(data=df, x=df.index, y='accuracy', ax=axes[0], label=stream_name, legend=False)
        handles.append(line.lines[0])
        labels.append(stream_name)

        sns.lineplot(data=df, x=df.index, y='emissions', ax=axes[1], label=stream_name, legend=False)

    axes[0].set_title('Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    axes[1].set_title('Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    plt.suptitle(title)
    # plt.figlegend(handles, labels, loc = 'lower center', ncol=3, labelspacing=0.)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [list(set(sum(lol, []))) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc ='lower center', ncols=9, bbox_to_anchor=(0.55, 0))

    plt.tight_layout()
    path = PLOT_DIR_BA / 'feature_comparison' / 'two_plots' / title
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'one_line_per_stream', bbox_inches='tight')
    plt.show()


def three_feature_plots_per_stream(df_per_stream: Dict[str, pd.DataFrame], feature: str, window_size: int) -> None:
    """Plot 3 metrics per stream for given feature and window_size"""
    sns.set_color_codes(COLOR_PALATE)
    # figure
    fig, axes = plt.subplots(ncols=3)
    handles, labels = [], []

    for stream_name, df in df_per_stream.items():
        df = df.rolling(window_size, center=True ,step=window_size).mean()
        line = sns.lineplot(data=df, x=df.index, y='accuracy', ax=axes[0], label=stream_name, legend=False)
        handles.append(line.lines[0])
        labels.append(stream_name)

        sns.lineplot(data=df, x=df.index, y='emissions', ax=axes[1], label=stream_name, legend=False)

        df['accuracy_per_emission'] = df['accuracy'] / df['emissions']
        sns.lineplot(data=df, x=df.index, y='accuracy_per_emission', ax=axes[2], label=stream_name, legend=False)

    axes[0].set_title('Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    axes[1].set_title('Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    axes[2].set_title('Accuracy / Emissions vs Instances')
    axes[2].set_xlabel('Number of Instances')
    axes[2].set_ylabel('$\\Delta\\left(\\frac{\\text{Accuracy}}{\\text{Emissions}}\\right)$')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    plt.suptitle(title)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [list(set(sum(lol, []))) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc ='lower center', ncols=9, bbox_to_anchor=(0.55, 0))

    plt.tight_layout()
    path = PLOT_DIR_BA / 'feature_comparison' / 'three_plots' / title
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'one_line_per_stream', bbox_inches='tight')
    plt.show()


def two_feature_plots(x: pd.DataFrame, feature: str, window_size: int) -> None:
    '''plot the mean of a feature in 3 ways in a single figure'''
    sns.set_color_codes(COLOR_PALATE)
    # data preparation:
    x.rolling(window_size, center=True ,step=window_size).mean()

    # figure
    fig, axes = plt.subplots(ncols=2)

    #  3 subplots
    sns.lineplot(x['accuracy'], ax=axes[0])
    axes[0].set_title('Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    sns.lineplot(x['emissions'], ax=axes[1])
    axes[1].set_title('Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    plt.suptitle(title)
    plt.tight_layout()

    path = PLOT_DIR_BA / 'feature_comparison' / 'two_plots' / title
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'mean_of_all_streams', bbox_inches='tight')
    plt.show()


def three_feature_plots_mean(x: pd.DataFrame, feature: str, window_size: int) -> None:
    '''plot the mean of a feature in 3 ways in a single figure'''
    sns.set_color_codes(COLOR_PALATE)
    # data preparation:
    x.rolling(window_size, center=True ,step=window_size).mean()

    # figure
    fig, axes = plt.subplots(ncols=3)

    #  3 subplots
    sns.lineplot(x['accuracy'], ax=axes[0])
    axes[0].set_title('Accuracy vs Instances')
    axes[0].set_xlabel('Number of Instances')
    axes[0].set_ylabel('$\\Delta$ Accuracy [%]')

    sns.lineplot(x['emissions'], ax=axes[1])
    axes[1].set_title('Emissions vs Instances')
    axes[1].set_xlabel('Number of Instances')
    axes[1].set_ylabel('$\\Delta$ Emissions [$kg\\ CO_2 \\text{equiv}$]')

    x['accuracy_per_emission'] = x['accuracy'] / (x['emissions'])
    sns.lineplot(data=x, x=x.index, y='accuracy_per_emission', ax=axes[2])
    axes[2].set_title('Accuracy / Emissions vs Instances')
    axes[2].set_xlabel('Number of Instances')
    axes[2].set_ylabel('$\\Delta\\left(\\frac{\\text{Accuracy}}{\\text{Emissions}}\\right)$')

    # layout options
    title = LEARNER_CONFIG_TO_NAMES[feature]
    plt.suptitle(title)
    plt.tight_layout()

    path = PLOT_DIR_BA / 'feature_comparison' / 'three_plots' / title 
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / 'mean_of_all_streams', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option('display.max_colwidth', None)
    logging.basicConfig(level=logging.INFO, filename=(PROJECT_FOLDER_PATH / 'ba_plot.log').as_posix())
    rename_folders(99)
    plot_feature_comparision()
    plot_hyperparameter_in_iso()
    # plot_hyperparameter_stable_vs_unstable()
