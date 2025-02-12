import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any, Set, Optional

import pandas as pd
from capymoa.datasets import ElectricityTiny
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import Stream

from ADLClassifier import ADLClassifier, global_grace_period, grace_period_per_layer, extend_classifier_for_evaluation, \
    winning_layer_training, vectorized_for_loop, BaseLearningRateProgression, disabeling_deleted_layers
from Evaluation.PlottingFunctions import __plot_and_save_result, __compare_all_of_one_run
from Evaluation.RayTuneResources._config import MAX_INSTANCES, ADWIN_DELTA_STANDIN


def __get_run_id() -> int:
    results_dir_path = Path("results/runs/")
    results_dir_path.mkdir(parents=True, exist_ok=True)
    if any(results_dir_path.iterdir()):
        *_, elem = results_dir_path.iterdir()
        name, running_nr = elem.stem.split("=")
        return int(running_nr) + 1
    else:
        return 1


def __evaluate_on_stream(
        stream_data: Stream,
        run_id: int,
        adl_parameters: Dict[str, Any],
        classifier: type(ADLClassifier),
        rename_values: Dict[str, float],
        stream_name: Optional[str] = None,
) -> None:
    print("--------------------------------------------------------------------------")
    print(f"---------------Start time: {datetime.now()}---------------------")


    adl_classifier = classifier(
        schema=stream_data.schema,
        **adl_parameters
    )

    assert hasattr(adl_classifier, "record_of_model_shape"), f"ADL classifier {adl_classifier} does not keep track of model shape, and cannot be evaluated"

    name_string_of_stream_data = stream_name if stream_name is not None else f"{stream_data._filename.split('.')[0]}/"
    hyperparameter_part_of_name_string = "_".join((f"{str(key).replace('_', ' ')}={str(value).replace('_', ' ')}" for key, value in rename_values.items()))
    results_dir_path = Path("results/runs")
    results_dir_path.mkdir(parents=True, exist_ok=True)

    results_path = results_dir_path / f"runID={run_id}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    results_path.mkdir(parents=True, exist_ok=True)

    os.environ["CODECARBON_OUTPUT_DIR"] = str(results_path)
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_LOG_LEVEL"] = "CRITICAL"

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES)
    total_time_end = time.time_ns()

    print(f"summary for training:\nrunId={run_id}\nstream={name_string_of_stream_data}\n" + "\n".join((f"{str(key).replace('_', ' ')}={str(value).replace('_', ' ')}" for key, value in rename_values.items())) + ":")
    print("--------------------------------------------------------------------------")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")

    print(f"\n\tThe cumulative results:")
    print(f"instances={results_ht.cumulative.metrics_dict()['instances']}, accuracy={results_ht.cumulative.metrics_dict()['accuracy']}")

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    # add custom parameters to result dict
    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end.insert(loc=0, column=key, value=[adl_classifier.record_of_model_shape[key][-1]])
        windowed_results[key] = adl_classifier.record_of_model_shape[key]
    metrics_at_end.insert(loc=0, column="overall time", value=((total_time_end-total_time_start) / 1e9))

    for key, val in rename_values.items():
        metrics_at_end.insert(loc=0, column=key, value=[str(val)])

    metrics_at_end.to_pickle(results_path / "metrics.pickle")
    windowed_results.to_pickle(results_path / "metrics_per_window.pickle")

    results_ht.write_to_file(results_path.absolute().as_posix())
    print(f"---------------End time: {datetime.now()}-----------------------")
    print("--------------------------------------------------------------------------")
    print()


def __write_summary(run_id: int, user_added_hyperparameter: Set[str]) -> None:
    runs_folder = Path(f"results/runs/runID={run_id}")
    summary = pd.DataFrame(
        columns=list(
            {
                "accuracy", "nr_of_layers", "instances", "overall time",
                "amount of active layers",
                "runID", "path", "stream",
                *user_added_hyperparameter
            }
        )
    )

    rename = {
        "nr_of_layers": "amount of hidden layers",
        "instances": "amount of instances",
        "lr": "learning rate",
        "runID": "run id",
        "MCICutOff": "mci"
    }
    i = 0
    for root, dirs, files in os.walk(runs_folder):
        if "metrics.pickle" in files:
            runIdStr, hyperparameter_string, stream_name = root.split("/")[2:]
            hyperparameter_dict_from_string = {key.replace(' ', '_'): value.replace(' ', '_') for key, value in [pair_string.split("=") for pair_string in hyperparameter_string.split("_")]}

            metrics = pd.read_pickle(Path(root) / "metrics.pickle")
            tmp_dict = {key: list(value.values())[0] for key, value in metrics.filter(summary.columns).to_dict().items()}
            tmp_dict.update(
                {
                    "stream": stream_name,
                    "runID": run_id,
                    "amount of active layers": len(metrics.loc[:, "active_layers"].iloc[0]),
                    "path": Path(root),
                 }
            )
            tmp_dict.update(hyperparameter_dict_from_string)
            summary.loc[i] = tmp_dict
            i += 1

    summary = summary.rename(columns=rename)
    runs_folder.mkdir(exist_ok=True, parents=True)
    order_of_columns = (
            [filtered_param if filtered_param not in rename else rename[filtered_param] for filtered_param in [added_parameter for added_parameter in user_added_hyperparameter if added_parameter != "classifier"]]
            + ["accuracy", "amount of hidden layers", "amount of active layers", "overall time", "classifier", "amount of instances", "run id", "path", "stream"]
                        )
    summary = summary.loc[:, order_of_columns]
    summary.to_csv(runs_folder / "summary.csv", sep="\t")


def _evaluate_parameters(
        adl_classifiers: List[type(ADLClassifier)],
        streams: List[Stream],
        learning_rates: Optional[List[Union[float | BaseLearningRateProgression]]] = None,
        mci_thresholds: Optional[List[float]] = None,
        adwin_deltas: Optional[List[float]] = None,
        grace_periods_for_layer: Optional[List[int]] = None,
        grace_periods_global: Optional[List[int]] = None,
):
    run_id = __get_run_id()
    added_hyperparameters = {"classifier", "stream"}
    start_time = time.time_ns()
    start = datetime.now()
    total_nr_of_runs = len(adl_classifiers) * len(streams) * len(learning_rates or [None]) * len(mci_thresholds or [None]) * len(adwin_deltas or [None]) * (max(1, len(grace_periods_for_layer or []) + len(grace_periods_global or [])))
    current_run_index = 1
    for classifier in adl_classifiers:
        values_of_renames = {
            "classifier": classifier.name()
        }
        for stream_data in streams:
            for lr in (learning_rates or [None]):
                for mci_threshold in (mci_thresholds or [None]):
                    for adwin_delta in (adwin_deltas or [None]):
                        added_parameters = {}
                        if adwin_delta is not None:
                            added_parameters["drift_detector"] = ADWIN(delta=adwin_delta)
                            values_of_renames[ADWIN_DELTA_STANDIN] = adwin_delta
                            added_hyperparameters.add(ADWIN_DELTA_STANDIN)
                        else:
                            added_parameters.pop('drift_detector', None)
                            values_of_renames.pop(ADWIN_DELTA_STANDIN, None)

                        if lr is not None:
                            added_parameters["lr"] = lr
                            values_of_renames["lr"] = lr
                            added_hyperparameters.add("lr")
                        else:
                            added_parameters.pop('lr', None)
                            values_of_renames.pop('lr', None)

                        if mci_threshold is not None:
                            added_parameters["mci_threshold_for_layer_pruning"] = mci_threshold
                            values_of_renames["MCICutOff"] = mci_threshold
                            added_hyperparameters.add("MCICutOff")
                        else:
                            added_parameters.pop("mci_threshold_for_layer_pruning", None)
                            values_of_renames.pop("MCICutOff", None)

                        for grace_period in (grace_periods_global or ([None] if grace_periods_for_layer is None else [])):
                            if grace_period is not None:
                                classifier_to_give = global_grace_period(grace_period)(classifier)
                                values_of_renames["globalGracePeriod"] = grace_period
                                added_hyperparameters.add("globalGracePeriod")
                            else:
                                classifier_to_give = classifier
                                values_of_renames.pop('globalGracePeriod', None)

                            print(f"---------------------------test: {current_run_index}/{total_nr_of_runs} = {current_run_index/total_nr_of_runs * 100 :.2f}%-----------------------------")
                            print(f"------running since: {start}, for {(time.time_ns() - start_time) / (1e9 * 60) :.2f}min = {(time.time_ns() - start_time) / (1e9 * 60**2) :.2f}h------")
                            print(f"---------------expected finish: {start + ((datetime.now() - start) * total_nr_of_runs/current_run_index)} ---------------")
                            __evaluate_on_stream(
                                stream_data=stream_data,
                                run_id=run_id,
                                classifier=classifier_to_give,
                                adl_parameters=added_parameters,
                                rename_values=values_of_renames
                            )
                            current_run_index += 1

                            values_of_renames.pop("globalGracePeriod", None)

                        for grace_period in grace_periods_for_layer or []:
                            if grace_period is not None:
                                classifier_to_give = grace_period_per_layer(grace_period)(classifier)
                                values_of_renames["gracePeriodPerLayer"] = grace_period
                                added_hyperparameters.add("gracePeriodPerLayer")
                            else:
                                classifier_to_give = classifier
                                values_of_renames.pop('gracePeriodPerLayer', None)
                            print(f"--------------------------test: {current_run_index}/{total_nr_of_runs} = {current_run_index/total_nr_of_runs * 100 :.2f}%-----------------------------")
                            print(f"------running since: {start}, for {(time.time_ns() - start_time) / (1e9 * 60) :.2f}min = {(time.time_ns() - start_time) / (1e9 * 60**2) :.2f}h------")
                            print(f"---------------expected finish: {start + ((datetime.now() - start) * total_nr_of_runs/current_run_index)} ---------------")
                            __evaluate_on_stream(
                                stream_data=stream_data,
                                run_id=run_id,
                                classifier=classifier_to_give,
                                adl_parameters=added_parameters,
                                rename_values=values_of_renames
                            )

                            current_run_index += 1
                            values_of_renames.pop("gracePeriodPerLayer", None)

    __write_summary(run_id, added_hyperparameters)

    __plot_and_save_result(run_id, show=False)

    if total_nr_of_runs <= 10:
        __compare_all_of_one_run(run_id, show=False)


def _test_example(name: Optional[str] = None):

    streams = [
        ElectricityTiny(),
        # simple_agraval_single_drift
        # Electricity()
    ]
    learning_rates = [
        # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.1),
        # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
        # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
        # LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.1),
        # LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.01),
        # LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.001),

        # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.1),
        # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
        # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
        # ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.1),
        # ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.01),
        # ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.001),
        5e-1,
        # 1e-1,
        # 5e-2,
        # 1e-2,
        # 1e-3
    ]

    mci_thresholds = [
        1e-5,
        # 1e-6, 
        # 1e-7,
        # 1e-8
    ]
    classifiers = [
        extend_classifier_for_evaluation(vectorized_for_loop),
        extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop),
        extend_classifier_for_evaluation(disabeling_deleted_layers, winning_layer_training, vectorized_for_loop)
        # extend_classifier_for_evaluation(winning_layer_training),
    ]

    adwin_deltas=[
        # 1e-1, 1e-2,
        # 1e-3,
        # 1e-4,
        # 1e-5,
        1e-6,
        # 1e-7,
        # 1e-8, 1e-9, 1e-10
    ]

    grace_periods_for_layer = [
        # None,
        4,
        # 8, 16,
        # 32
    ]
    grace_periods_global = None

    run_id = __get_run_id()

    _evaluate_parameters(
        adl_classifiers=classifiers,
        streams=streams,
        learning_rates=learning_rates,
        mci_thresholds=mci_thresholds,
        adwin_deltas=adwin_deltas,
        grace_periods_global=grace_periods_global,
        grace_periods_for_layer=grace_periods_for_layer,
    )

    if name is not None:
        folder = Path("/home/david/PycharmProjects/ADL/results/experiment_data_selected") / name
        run_folder = Path(f"/home/david/PycharmProjects/ADL/results/runs/runID={run_id}")
        comparision_folder = Path("/home/david/PycharmProjects/ADL/results/comparisons/comparison=0")

        shutil.move(run_folder, folder)
        shutil.move(comparision_folder, folder)