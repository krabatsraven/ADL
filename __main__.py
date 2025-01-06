import os
import time
from functools import reduce
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from capymoa.stream import Stream
from torch.onnx import export

from ADLClassifier import ADLClassifier

from capymoa.datasets import Electricity, ElectricityTiny
from capymoa.evaluation import prequential_evaluation


def __evaluate_on_stream(stream_data: Stream, learning_rate: float, threshold_for_layer_pruning: float, classifier: type(ADLClassifier) = ADLClassifier) -> None:
    adl_classifier = classifier(schema=stream_data.schema, lr=learning_rate, mci_threshold_for_layer_pruning=threshold_for_layer_pruning)

    name_string_of_stream_data = f"{stream_data._filename.split('.')[0]}/"
    hyperparameter_part_of_name_string = f"lr={learning_rate}_MCICutOff={threshold_for_layer_pruning}_classifier={adl_classifier}"
    results_dir_path = Path("results/")
    results_dir_path.mkdir(parents=True, exist_ok=True)
    if any(results_dir_path.iterdir()):
        *_, elem = results_dir_path.iterdir()
        name, running_nr = elem.stem.split("=")
        results_path = results_dir_path / f"{name}={int(running_nr) + 1}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    else:
        results_path = results_dir_path / "runID=1" / hyperparameter_part_of_name_string / name_string_of_stream_data

    results_path.mkdir(parents=True, exist_ok=True)
    os.environ["CODECARBON_OUTPUT_DIR"] = str(results_path)
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_LOG_LEVEL"] = "CRITICAL"

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier, window_size=100, optimise=True, store_predictions=False, store_y=False)
    total_time_end = time.time_ns()

    # todo: delete print statement and time to train as it measures part of the evaluating process and will become obsolete shortly:
    print(stream_data._filename)
    print(f"total time spend in covariance loop: {adl_classifier.total_time_in_loop:.2E}ns, that equals {adl_classifier.total_time_in_loop / 10 ** 9:.2f}s or {adl_classifier.total_time_in_loop / 10 ** 9 /60:.2}min")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"meaning that the covariance loop alone took {adl_classifier.total_time_in_loop / (total_time_end - total_time_start) * 100}% of the training time")

    print(f"\n\tAll the cumulative results:")
    print(results_ht.cumulative.metrics_dict())

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    # add custom parameters to result dict
    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end[key] = str(adl_classifier.record_of_model_shape[key][-1])
        windowed_results[key] = adl_classifier.record_of_model_shape[key]

    metrics_at_end.to_pickle(results_path / "metrics.pickle")
    windowed_results.to_pickle(results_path / "metrics_per_window.pickle")

    results_ht.write_to_file(results_path.absolute().as_posix())

def __ax_pretty(
        sns_ax,
        target_path,
        title_string="",
        sub_title_string="",
        show=True,
        special_x_label=None,
        special_y_label=None
):
    plt.suptitle(title_string)
    plt.title(sub_title_string)
    if special_x_label is not None:
        sns_ax.set_xlabels(special_x_label)
    else:
        sns_ax.set_xlabels(sns_ax.ax.get_xlabel().replace("_", " ").title())
    if special_y_label is not None:
        sns_ax.set_ylabels(special_y_label)
    else:
        sns_ax.set_ylabels(sns_ax.ax.get_ylabel().replace("_", " ").title())
    sns_ax.tight_layout()
    plt.savefig(target_path)
    if show:
        plt.show()


def __compare_results_via_plot_and_save(result_paths: List[Path]) -> None:
    working_folder = Path("results") / "comparisons" / "_VS_".join(map(lambda f: f"{f.parent.parent.name}_{f.parent.name}", result_paths))
    plot_folder = working_folder / "plots"
    plot_folder.mkdir(exist_ok=True, parents=True)

    dfs = []
    df_names = []
    df_contains_emission_flags = []
    hyperparameters = []

    for i, result_dir in enumerate(result_paths):
        metrics_windowed_path, _, *tail = [file for file in result_dir.iterdir() if file.is_file()].__reversed__()
        run_data = pd.read_pickle(metrics_windowed_path)
        contains_emissions = False
        if tail:
            emissions = pd.read_csv(tail[0])
            run_data = run_data.merge(emissions, right_index=True, left_index=True)
            contains_emissions = True

        run_data.name = f"TABLE{i}"
        df_names.append(f"{result_dir.parent.parent.name}+{result_dir.parent.name.replace('_', '+')}+{result_dir.name}")

        dfs.append(run_data)
        df_contains_emission_flags.append(contains_emissions)

        hyperparameter_folder = result_dir.parent.resolve()
        lr, mci_cut, *_ = map(lambda s: s.split("=")[1], hyperparameter_folder.name.split("_"))
        lr, mci_cut = float(lr), float(mci_cut)
        hyperparameters.append((lr, mci_cut))

    sub_title_string = ""
    title_string = ""

    def merge_two_dfs(lhs, rhs):
        out = pd.merge(lhs, rhs, on="instances", how="outer", suffixes=("_" + lhs.name, "_" + rhs.name))
        out.name = rhs.name
        return out

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_seq_items", None)

    data = reduce(merge_two_dfs, dfs)

    plt.show()

    _ax_pretty = lambda *args, **kwargs: __ax_pretty(*args, **kwargs, title_string=title_string, sub_title_string=sub_title_string)
   
    title_string = "Instances trained on vs Accuracy".title()

    data_filtered = data.filter(regex="instances|accuracy_.*")
    dfl = pd.melt(data_filtered, ['instances'])
    # todo: rename variabble to difference in table names

    ax = sns.relplot(data=dfl, x="instances", y="value", hue="variable", s=10)
    _ax_pretty(ax, (plot_folder / title_string))
    
    # todo: use above on every graph choice
    # title_string = "Amount of active output layers vs Accuracy".title()
    # amount_of_active_layers=run_data.active_layers.apply(len)
    # ax = sns.relplot(x=amount_of_active_layers, y=run_data.accuracy, s=10)
    # _ax_pretty(ax, (plot_folder / title_string))
    # 
    # title_string = "Amount of Hidden Layers vs Accuracy".title()
    # ax = sns.relplot(x=run_data.nr_of_layers, y=run_data.accuracy, s=10)
    # _ax_pretty(ax, (plot_folder / title_string), special_x_label="Amount of Hidden Layers")
    # 
    # title_string = "Amount of Hidden Nodes vs Accuracy".title()
    # amount_of_hidden_nodes = run_data.shape_of_hidden_layers.apply(lambda lst: sum((tpl[1] for tpl in lst)))
    # ax = sns.relplot(x=amount_of_hidden_nodes, y=run_data.accuracy, s=10)
    # _ax_pretty(ax, (plot_folder / title_string), special_x_label="Amount of Nodes in Hidden Layers")
    # 
    # title_string = "Instances vs Amount of Hidden Layers".title()
    # ax = sns.relplot(x=run_data.instances, y=run_data.nr_of_layers, s=10)
    # _ax_pretty(ax, (plot_folder / title_string), special_y_label="Amount of Hidden Layers")
    # 
    # title_string = "Instances vs Amount of Hidden Nodes".title()
    # ax = sns.relplot(x=run_data.instances, y=amount_of_hidden_nodes, s=10)
    # _ax_pretty(ax, (plot_folder/title_string), special_y_label="Amount of Nodes in Hidden Layers")
    # 
    # title_string = "Instances vs Amount of Active Layers"
    # ax = sns.relplot(x=run_data.instances, y=amount_of_active_layers, s=10)
    # _ax_pretty(ax, (plot_folder/title_string), special_y_label="Amount of Active Layers")
    # 
    # if all(df_contains_emission_flags):
    #     title_string = "Instances vs Emissions".title()
    #     ax = sns.relplot(x=run_data.instances, y=run_data.emissions, s=10)
    #     _ax_pretty(ax, (plot_folder/title_string))
    # 
    #     title_string = "Amount of Active Layers vs Emissions".title()
    #     ax = sns.relplot(x=amount_of_active_layers, y=run_data.emissions, s=10)
    #     _ax_pretty(ax, (plot_folder/title_string))
    # 
    #     title_string = "nr of nodes vs emissions".title()
    #     ax = sns.relplot(x=amount_of_hidden_nodes, y=run_data.emissions, s=10)
    #     _ax_pretty(ax, (plot_folder/title_string), special_x_label="Amount of Nodes in Hidden Layers")


def __plot_and_save_result(result_id: int, show: bool=True) -> None:
    results_dir_path = Path(f"results/runID={result_id}")
    if not results_dir_path.exists():
        print(f"runID={result_id}: No results found, returning")
        return
    hyperparameter_folders = list(results_dir_path.iterdir())
    if not any(hyperparameter_folders):
        print(f"runID={result_id}: No test run with hyperparameters found, returning")
        return

    for hyperparameter_folder in hyperparameter_folders:
        datastream_folders = list(hyperparameter_folder.iterdir())
        if not any(datastream_folders):
            print(f"runID={result_id}: No test run with a datastream found for hyperparameter={hyperparameter_folder.name}, skipping")
            continue

        for datastream_folder in datastream_folders:
            metrics_overview_path = datastream_folder / "metrics.pickle"

            if not metrics_overview_path.exists():
                print(f"runID={result_id}: No metrics overview file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            metrics_overview = pd.read_pickle(metrics_overview_path)

            all_metrics_path = datastream_folder / "metrics_per_window.pickle"
            if not all_metrics_path.exists():
                print(f"runID={result_id}: No metrics file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            results_csv = pd.read_pickle(all_metrics_path)
            emissions_path = datastream_folder / "emissions.csv"
            emissions_plotting = False
            if emissions_path.exists():
                emissions = pd.read_csv(emissions_path)
                results_csv = results_csv.merge(emissions, right_index=True, left_index=True)
                emissions_plotting = True

            # plotting
            lr, mci_cut, *_ = map(lambda s: s.split("=")[1], hyperparameter_folder.name.split("_"))
            lr, mci_cut = float(lr), float(mci_cut)
            sub_title_string = f"instances={metrics_overview.instances[0]:n}, lr={lr:.2e}, mci-cut-off={mci_cut :.2e},\nmean-accuracy={metrics_overview.accuracy[0]:.2f},\nnr of active layers after training={metrics_overview.active_layers.apply(len)[0]:n}".title()
            plot_folder = datastream_folder / "plots"
            plot_folder.mkdir(exist_ok=True)
            title_string=""

            _ax_pretty = lambda *args, **kwargs: __ax_pretty(*args, **kwargs, title_string=title_string, sub_title_string=sub_title_string, show=show)

            title_string = "Instances trained on vs Accuracy".title()
            ax = sns.relplot(x=results_csv.instances, y=results_csv.accuracy, s=10)
            _ax_pretty(ax, (plot_folder / title_string))

            title_string = "Amount of active output layers vs Accuracy".title()
            amount_of_active_layers=results_csv.active_layers.apply(len)
            ax = sns.relplot(x=amount_of_active_layers, y=results_csv.accuracy, s=10)
            _ax_pretty(ax, (plot_folder / title_string))

            title_string = "Amount of Hidden Layers vs Accuracy".title()
            ax = sns.relplot(x=results_csv.nr_of_layers, y=results_csv.accuracy, s=10)
            _ax_pretty(ax, (plot_folder / title_string), special_x_label="Amount of Hidden Layers")

            title_string = "Amount of Hidden Nodes vs Accuracy".title()
            amount_of_hidden_nodes = results_csv.shape_of_hidden_layers.apply(lambda lst: sum((tpl[1] for tpl in lst)))
            ax = sns.relplot(x=amount_of_hidden_nodes, y=results_csv.accuracy, s=10)
            _ax_pretty(ax, (plot_folder / title_string), special_x_label="Amount of Nodes in Hidden Layers")

            title_string = "Instances vs Amount of Hidden Layers".title()
            ax = sns.relplot(x=results_csv.instances, y=results_csv.nr_of_layers, s=10)
            _ax_pretty(ax, (plot_folder / title_string), special_y_label="Amount of Hidden Layers")

            title_string = "Instances vs Amount of Hidden Nodes".title()
            ax = sns.relplot(x=results_csv.instances, y=amount_of_hidden_nodes, s=10)
            _ax_pretty(ax, (plot_folder / title_string), special_y_label="Amount of Nodes in Hidden Layers")

            title_string = "Instances vs Amount of Active Layers"
            ax = sns.relplot(x=results_csv.instances, y=amount_of_active_layers, s=10)
            _ax_pretty(ax, (plot_folder / title_string), special_y_label="Amount of Active Layers")

            if emissions_plotting:
                title_string = "Instances vs Emissions".title()
                ax = sns.relplot(x=results_csv.instances, y=results_csv.emissions, s=10)
                _ax_pretty(ax, (plot_folder / title_string))

                title_string = "Amount of Active Layers vs Emissions".title()
                ax = sns.relplot(x=amount_of_active_layers, y=results_csv.emissions, s=10)
                _ax_pretty(ax, (plot_folder / title_string))

                title_string = "nr of nodes vs emissions".title()
                ax = sns.relplot(x=amount_of_hidden_nodes, y=results_csv.emissions, s=10)
                _ax_pretty(ax, (plot_folder / title_string), special_x_label="Amount of Nodes in Hidden Layers")


if __name__ == "__main__":

    run = False

    if run:
        streams = [ElectricityTiny()]
        learning_rates = [1e-3]
        mci_thresholds = [1e-7]

        for stream_data in streams:
            for lr in learning_rates:
                for mci_threshold in mci_thresholds:
                    __evaluate_on_stream(stream_data=stream_data, learning_rate=lr, threshold_for_layer_pruning=mci_threshold)

    # plot_and_save_result(1)
    # plot_and_save_result(2)
    # plot_and_save_result(3)
    # plot_and_save_result(4)
    # plot_and_save_result(5)
    # plot_and_save_result(6)
    # __plot_and_save_result(9)


    run_7_7_8_and_9 = [
        Path("/home/david/PycharmProjects/ADL/results/runID=7/lr=0.001_MCICutOff=1e-07/electricity_tiny"),
        Path("/home/david/PycharmProjects/ADL/results/runID=7/lr=0.001_MCICutOff=1e-07/electricity_tiny"),
        Path("/home/david/PycharmProjects/ADL/results/runID=8/lr=0.001_MCICutOff=1e-07/electricity_tiny"),
        Path("/home/david/PycharmProjects/ADL/results/runID=8/lr=0.001_MCICutOff=1e-07/electricity_tiny")
    ]
    __compare_results_via_plot_and_save(run_7_7_8_and_9)