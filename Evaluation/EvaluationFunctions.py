import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union, Any, Set, Optional

import pandas as pd
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import Stream

from ADLClassifier import ADLClassifier, global_grace_period, grace_period_per_layer
from Evaluation.PlottingFunctions import __plot_and_save_result, __compare_all_of_one_run

ADWIN_DELTA_STANDIN = "adwin-delta"

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
        rename_values: Dict[str, float]
) -> None:
    adl_classifier = classifier(
        schema=stream_data.schema,
        **adl_parameters
    )

    assert hasattr(adl_classifier, "record_of_model_shape"), f"ADL classifier {adl_classifier} does not keep track of model shape, and cannot be evaluated"

    name_string_of_stream_data = f"{stream_data._filename.split('.')[0]}/"
    hyperparameter_part_of_name_string = "" # f"classifier={adl_classifier.__str__()}"
    user_added_parameters_string = "_".join((f"{str(key).replace('_', ' ')}={str(value).replace('_', ' ')}" for key, value in rename_values.items()))
    if len(user_added_parameters_string) > 0:
        hyperparameter_part_of_name_string = user_added_parameters_string #+ "_" + hyperparameter_part_of_name_string
    results_dir_path = Path("results/runs")
    results_dir_path.mkdir(parents=True, exist_ok=True)

    results_path = results_dir_path / f"runID={run_id}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    results_path.mkdir(parents=True, exist_ok=True)

    os.environ["CODECARBON_OUTPUT_DIR"] = str(results_path)
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_LOG_LEVEL"] = "CRITICAL"

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier, window_size=1, optimise=True, store_predictions=False, store_y=False)
    total_time_end = time.time_ns()

    print(stream_data._filename)
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")

    print(f"\n\tAll the cumulative results:")
    print(results_ht.cumulative.metrics_dict())

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    # add custom parameters to result dict
    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end.insert(loc=0, column=key, value=[adl_classifier.record_of_model_shape[key][-1]])
        windowed_results[key] = adl_classifier.record_of_model_shape[key]
    metrics_at_end.insert(loc=0, column="overall time", value=((total_time_end-total_time_start) / 1e9))

    for key, val in rename_values.items():
        metrics_at_end.insert(loc=0, column=key, value=[val])

    metrics_at_end.to_pickle(results_path / "metrics.pickle")
    windowed_results.to_pickle(results_path / "metrics_per_window.pickle")

    results_ht.write_to_file(results_path.absolute().as_posix())


def __write_summary(run_id: int, user_added_hyperparameter: Set[str]) -> None:
    runs_folder = Path(f"results/runs/runID={run_id}")
    summary = pd.DataFrame(
        columns=list(
            {
                "accuracy", "nr_of_layers", "instances", "overall time",
                "amount of active layers",
                "runID", "path",
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
    summary.to_csv(runs_folder / "summary.csv", sep="\t")


def _evaluate_parameters(
        adl_classifiers: List[type(ADLClassifier)], 
        streams: List[Stream],
        learning_rates: Optional[List[float]] = None,
        mci_thresholds: Optional[List[float]] = None,
        adwin_deltas: Optional[List[float]] = None,
        grace_periods_for_layer: Optional[List[int]] = None,
        grace_periods_global: Optional[List[int]] = None,
):
    run_id = __get_run_id()
    added_hyperparameters = {"classifier", "stream"}
    for classifier in adl_classifiers:
        classifier_to_give = classifier
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
                        if lr is not None:
                            added_parameters["lr"] = lr
                        if mci_threshold is not None:
                            added_parameters["mci_threshold_for_layer_pruning"] = mci_threshold
                            values_of_renames["MCICutOff"] = mci_threshold
                            added_hyperparameters.add("MCICutOff")
                        for grace_period in (grace_periods_global or [None]):
                            if grace_period is not None:
                                classifier_to_give = global_grace_period(grace_period)(classifier)
                                values_of_renames["globalGracePeriod"] = grace_period
                                added_hyperparameters.add("globalGracePeriod")
                            __evaluate_on_stream(
                                stream_data=stream_data,
                                run_id=run_id,
                                classifier=classifier,
                                adl_parameters=added_parameters,
                                rename_values=values_of_renames
                            )

                        values_of_renames.pop("globalGracePeriod", None)

                        for grace_period in grace_periods_for_layer if grace_periods_for_layer is not None else [] if grace_periods_global is None else [None]:
                            if grace_period is not None:
                                classifier_to_give = grace_period_per_layer(grace_period)(classifier)
                                values_of_renames["gracePeriodPerLayer"] = grace_period
                                added_hyperparameters.add("gracePeriodPerLayer")
                            __evaluate_on_stream(
                                stream_data=stream_data,
                                run_id=run_id,
                                classifier=classifier_to_give,
                                adl_parameters=added_parameters,
                                rename_values=values_of_renames
                            )

    __write_summary(run_id, added_hyperparameters)

    __plot_and_save_result(run_id, show=False)
    __compare_all_of_one_run(run_id, show=False)

