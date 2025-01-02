import time
from pathlib import Path

import pandas as pd
from capymoa.stream import Stream

from ADLClassifier import ADLClassifier

from capymoa.datasets import Electricity, ElectricityTiny
from capymoa.evaluation import prequential_evaluation

def evaluate_on_stream(stream_data: Stream, learning_rate: float, threshold_for_layer_pruning: float) -> None:
    adl_classifier = ADLClassifier(schema=stream_data.schema, lr=learning_rate, mci_threshold_for_layer_pruning=threshold_for_layer_pruning)

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

    name_string_of_stream_data = f"{stream_data._filename.split('.')[0]}/"
    hyperparameter_part_of_name_string = f"lr={learning_rate}_MCICutOff={threshold_for_layer_pruning}"
    results_dir_path = Path("results/")
    results_dir_path.mkdir(parents=True, exist_ok=True)
    if any(results_dir_path.iterdir()):
        *_, elem = results_dir_path.iterdir()
        name, running_nr = elem.stem.split("=")
        results_path = results_dir_path / f"{name}={int(running_nr) + 1}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    else:
        results_path = results_dir_path / "runID=1" / hyperparameter_part_of_name_string / name_string_of_stream_data

    results_path.mkdir(parents=True, exist_ok=True)

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    # add custom parameters to result dict
    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end[key] = str(adl_classifier.record_of_model_shape[key][-1])
        windowed_results[key] = adl_classifier.record_of_model_shape[key]

    metrics_at_end.to_csv(results_path / "metrics.csv")
    windowed_results.to_csv(results_path / "metrics_per_window.csv")

    results_ht.write_to_file(results_path.absolute().as_posix())

def plot_and_save_result(result_id: int) -> None:
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
            metrics_overview_path = datastream_folder / "metrics.csv"

            if not metrics_overview_path.exists():
                print(f"runID={result_id}: No metrics overview file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            metrics_overview = pd.read_csv(metrics_overview_path)
            print(metrics_overview.columns)

            all_metrics_path = datastream_folder / "metrics_per_window.csv"
            if not all_metrics_path.exists():
                print(f"runID={result_id}: No metrics file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            results_csv = pd.read_csv(all_metrics_path)
            print(results_csv.columns)

if __name__ == "__main__":

    no_run = True

    if not no_run:
        streams = [ElectricityTiny()]
        learning_rates = [1e-3]
        mci_thresholds = [1e-7]

        for stream_data in streams:
            for lr in learning_rates:
                for mci_threshold in mci_thresholds:
                    evaluate_on_stream(stream_data=stream_data, learning_rate=lr, threshold_for_layer_pruning=mci_threshold)

    plot_and_save_result(1)
    plot_and_save_result(2)
    plot_and_save_result(3)
    plot_and_save_result(4)
    plot_and_save_result(5)
    plot_and_save_result(6)
