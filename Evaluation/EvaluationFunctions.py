import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any, Set, Optional

import pandas as pd
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import Stream

from ADLClassifier import ADLClassifier, global_grace_period, grace_period_per_layer, extend_classifier_for_evaluation, \
    winning_layer_training, vectorized_for_loop, BaseLearningRateProgression, disabeling_deleted_layers, \
    delete_deleted_layers, input_preprocessing, add_weight_correction_parameter_to_user_choices
from Evaluation import config_handling
from Evaluation.PlottingFunctions import __plot_and_save_result, __compare_all_of_one_run
from Evaluation._config import ADWIN_DELTA_STANDIN, MAX_INSTANCES_TEST, STREAM_STRINGS
from Evaluation.config_handling import get_best_config_for_stream_name, adl_run_data_from_config


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



    name_string_of_stream_data = stream_name if stream_name is not None else f"{stream_data.schema.dataset_name}/"
    hyperparameter_part_of_name_string = "_".join((f"{str(key).replace('_', ' ')}={f'{value:.3g}' if type(value) == float else str(value).replace('_', ' ')}" for key, value in rename_values.items()))
    results_dir_path = Path("results/runs")
    results_dir_path.mkdir(parents=True, exist_ok=True)

    results_path = results_dir_path / f"runID={run_id}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    results_path.mkdir(parents=True, exist_ok=True)

    os.environ["CODECARBON_OUTPUT_DIR"] = str(results_path)
    # os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_LOG_LEVEL"] = "CRITICAL"

    print(f"summary for training:\nrunId={run_id}\nstream={name_string_of_stream_data}\n" + "\n".join((f"{str(key).replace('_', ' ')}={str(value).replace('_', ' ')}" for key, value in rename_values.items())) + ":")
    print("--------------------------------------------------------------------------")

    adl_classifier = classifier(
        schema=stream_data.schema,
        **adl_parameters
    )

    assert hasattr(adl_classifier, "record_of_model_shape"), f"ADL classifier {adl_classifier} does not keep track of model shape, and cannot be evaluated"

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES_TEST)
    total_time_end = time.time_ns()

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
    if (runs_folder / "summary.csv").exists():
        previous_summary = pd.read_csv(runs_folder / "summary.csv", sep="\t")
    else:
        previous_summary = None

    summary = pd.DataFrame(
        columns=list(
            {
                "accuracy", "nr_of_layers", "instances", "overall time",
                "amount of active layers", 'amount of nodes',
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
                    'amount of nodes': metrics.loc[:, "shape_of_hidden_layers"].map(lambda list_of_shapes: sum((layer_shape[1] for layer_shape in list_of_shapes))).iloc[0],
                    "path": Path(root),
                 }
            )
            tmp_dict.update(hyperparameter_dict_from_string)
            summary.loc[i] = tmp_dict
            i += 1

    summary = summary.rename(columns=rename)
    if previous_summary is not None:
        summary = pd.concat([previous_summary, summary]).reset_index()
    runs_folder.mkdir(exist_ok=True, parents=True)
    order_of_columns = (
            [filtered_param if filtered_param not in rename else rename[filtered_param] for filtered_param in [added_parameter for added_parameter in user_added_hyperparameter if added_parameter != "classifier"]]
            + ["accuracy", "amount of hidden layers", "amount of active layers", 'amount of nodes', "overall time", "classifier", "amount of instances", "run id", "path", "stream"]
                        )
    summary = summary.loc[:, order_of_columns]
    summary.to_csv(runs_folder / "summary.csv", sep="\t")


def _evaluate_parameters(
        adl_classifiers: List[type(ADLClassifier)],
        streams: List[Stream],
        learning_rates: Optional[List[Union[float | BaseLearningRateProgression]]] = None,
        learning_rate_for_weights: Optional[List[Union[float | BaseLearningRateProgression]]] = None,
        mci_thresholds: Optional[List[float]] = None,
        adwin_deltas: Optional[List[float]] = None,
        grace_periods_for_layer: Optional[List[int]] = None,
        grace_periods_global: Optional[List[int]] = None,
        stream_names: Optional[List[str]] = None,
):
    if stream_names is not None:
        assert len(stream_names) == len(streams), "give me as many names as streams or give me none"
        streams = list(zip(streams, stream_names))
    else:
        streams = list(zip(streams, [None] * len(streams)))
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
        classifier_wo_grace = classifier
        for stream_data, stream_name in streams:
            for lr in (learning_rates or [None]):
                for lr_weight in learning_rate_for_weights or [None]:
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

                            if lr_weight is not None:
                                added_parameters['layer_weight_learning_rate'] = lr_weight
                                values_of_renames['layerWeightLR'] = lr_weight
                                added_hyperparameters.add('layerWeightLR')
                                classifier_wo_grace = add_weight_correction_parameter_to_user_choices(classifier_wo_grace)
                            else:
                                added_parameters.pop('layer_weight_learning_rate', None)
                                values_of_renames.pop('layerWeightLR', None)
    
                            if mci_threshold is not None:
                                added_parameters["mci_threshold_for_layer_pruning"] = mci_threshold
                                values_of_renames["MCICutOff"] = mci_threshold
                                added_hyperparameters.add("MCICutOff")
                            else:
                                added_parameters.pop("mci_threshold_for_layer_pruning", None)
                                values_of_renames.pop("MCICutOff", None)
    
                            for grace_period in (grace_periods_global or ([None] if grace_periods_for_layer is None else [])):
                                if grace_period is not None:
                                    classifier_to_give = global_grace_period(grace_period)(classifier_wo_grace)
                                    values_of_renames["globalGracePeriod"] = grace_period
                                    added_hyperparameters.add("globalGracePeriod")
                                else:
                                    classifier_to_give = classifier_wo_grace
                                    values_of_renames.pop('globalGracePeriod', None)
    
                                print(f"---------------------------test: {current_run_index}/{total_nr_of_runs} = {current_run_index/total_nr_of_runs * 100 :.2f}%-----------------------------")
                                print(f"------running since: {start}, for {(time.time_ns() - start_time) / (1e9 * 60) :.2f}min = {(time.time_ns() - start_time) / (1e9 * 60**2) :.2f}h------")
                                print(f"---------------expected finish: {start + ((datetime.now() - start) * total_nr_of_runs/current_run_index)} ---------------")
                                __evaluate_on_stream(
                                    stream_data=stream_data,
                                    run_id=run_id,
                                    classifier=classifier_to_give,
                                    adl_parameters=added_parameters,
                                    rename_values=values_of_renames,
                                    stream_name=stream_name
                                )
                                current_run_index += 1
    
                                values_of_renames.pop("globalGracePeriod", None)
    
                            for grace_period in grace_periods_for_layer or []:
                                if grace_period is not None:
                                    classifier_to_give = grace_period_per_layer(grace_period)(classifier_wo_grace)
                                    values_of_renames["gracePeriodPerLayer"] = grace_period
                                    added_hyperparameters.add("gracePeriodPerLayer")
                                else:
                                    classifier_to_give = classifier_wo_grace
                                    values_of_renames.pop('gracePeriodPerLayer', None)
                                print(f"--------------------------test: {current_run_index}/{total_nr_of_runs} = {current_run_index/total_nr_of_runs * 100 :.2f}%-----------------------------")
                                print(f"------running since: {start}, for {(time.time_ns() - start_time) / (1e9 * 60) :.2f}min = {(time.time_ns() - start_time) / (1e9 * 60**2) :.2f}h------")
                                print(f"---------------expected finish: {start + ((datetime.now() - start) * total_nr_of_runs/current_run_index)} ---------------")
                                __evaluate_on_stream(
                                    stream_data=stream_data,
                                    run_id=run_id,
                                    classifier=classifier_to_give,
                                    adl_parameters=added_parameters,
                                    rename_values=values_of_renames,
                                    stream_name=stream_name
                                )
    
                                current_run_index += 1
                                values_of_renames.pop("gracePeriodPerLayer", None)

    __write_summary(run_id, added_hyperparameters)

    __plot_and_save_result(run_id, show=False)

    if total_nr_of_runs <= 10:
        __compare_all_of_one_run(run_id, show=False)


def _test_example(name: Optional[str] = None, with_co_2: bool = False):

    streams = list(map(config_handling.config_to_stream, STREAM_STRINGS))
    stream_names = STREAM_STRINGS
    learning_rate_for_weights = [0.001]
    learning_rates = [0.05]

    mci_thresholds = [1e-7]
    classifiers = [
        extend_classifier_for_evaluation(input_preprocessing, winning_layer_training, vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(delete_deleted_layers, input_preprocessing, winning_layer_training, vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(disabeling_deleted_layers, input_preprocessing, winning_layer_training, vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, winning_layer_training, vectorized_for_loop, with_emissions=with_co_2),
        extend_classifier_for_evaluation(vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, vectorized_for_loop, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, winning_layer_training, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, with_emissions=with_co_2),
        extend_classifier_for_evaluation(vectorized_for_loop, with_emissions=with_co_2),
        extend_classifier_for_evaluation(add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(winning_layer_training, with_emissions=with_co_2)
    ]

    adwin_deltas=[1e-7]

    grace_periods_for_layer = [4000]
    grace_periods_global = None

    run_id = __get_run_id()

    _evaluate_parameters(
        adl_classifiers=classifiers,
        streams=streams,
        learning_rates=learning_rates,
        learning_rate_for_weights=learning_rate_for_weights,
        mci_thresholds=mci_thresholds,
        adwin_deltas=adwin_deltas,
        grace_periods_global=grace_periods_global,
        grace_periods_for_layer=grace_periods_for_layer,
        stream_names=stream_names
    )

    if name is not None:
        folder = Path("/home/david/PycharmProjects/ADL/results/experiment_data_selected") / name
        run_folder = Path(f"/home/david/PycharmProjects/ADL/results/runs/runID={run_id}")
        comparision_folder = Path("/home/david/PycharmProjects/ADL/results/comparisons/comparison=0")

        if folder.exists():
            shutil.rmtree(folder)
        shutil.move(run_folder, folder)
        shutil.move(comparision_folder, folder)


def _test_best_combination(name: Optional[str] = None, with_co_2: bool = False):
    streams = list(map(config_handling.config_to_stream, STREAM_STRINGS))
    nr_of_combinations = len(streams)
    stream_names = STREAM_STRINGS
    # best_config = list(map(get_best_config_for_stream_name, STREAM_STRINGS))

    standard_config = {
        'lr': 0.190064,
        'learner': extend_classifier_for_evaluation(),
        'layer_weight_learning_rate': 0.153329,
        'adwin-delta': 0.000239152,
        'mci': 3.50888e-07,
        'grace_period': (128, 'layer_grace'),
        'loss_fn': 'NLLLoss'
    }
    best_config = [standard_config] * nr_of_combinations

    classifiers = [
        extend_classifier_for_evaluation(input_preprocessing, vectorized_for_loop, winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(delete_deleted_layers, input_preprocessing, vectorized_for_loop, winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(disabeling_deleted_layers, input_preprocessing, vectorized_for_loop, winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(vectorized_for_loop, winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, vectorized_for_loop, winning_layer_training, with_emissions=with_co_2),
        extend_classifier_for_evaluation(vectorized_for_loop, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(winning_layer_training, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation( vectorized_for_loop, winning_layer_training, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, vectorized_for_loop, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, winning_layer_training, with_emissions=with_co_2),
        extend_classifier_for_evaluation(input_preprocessing, with_emissions=with_co_2),
        extend_classifier_for_evaluation(vectorized_for_loop, with_emissions=with_co_2),
        extend_classifier_for_evaluation(add_weight_correction_parameter_to_user_choices, with_emissions=with_co_2),
        extend_classifier_for_evaluation(winning_layer_training, with_emissions=with_co_2)
    ]

    run_id = 43

    for i in range(nr_of_combinations):
        for classifier in classifiers:
            print(classifier.name())
            adl_parameter, rename_values, added_names = adl_run_data_from_config(best_config[i], with_weight_lr=('WithUserChosenWeightLR' in classifier.name()), with_co2=with_co_2, learner_name=classifier.name())
            __evaluate_on_stream(
                classifier=classifier,
                stream_data=streams[i],
                stream_name=stream_names[i],
                adl_parameters=adl_parameter,
                rename_values=rename_values,
                run_id=run_id
            )
            __write_summary(run_id, added_names)
            __plot_and_save_result(run_id, show=False)
        return

    if name is not None:
        folder = Path("/home/david/PycharmProjects/ADL/results/experiment_data_selected") / name
        run_folder = Path(f"/home/david/PycharmProjects/ADL/results/runs/runID={run_id}")
        comparision_folder = Path("/home/david/PycharmProjects/ADL/results/comparisons/comparison=0")

        if folder.exists():
            shutil.rmtree(folder)
        shutil.move(run_folder, folder)
        shutil.move(comparision_folder, folder)
