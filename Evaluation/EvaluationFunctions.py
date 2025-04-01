import logging
import os
import shutil
import time
from datetime import datetime
from filecmp import dircmp
from pathlib import Path
from typing import Dict, List, Union, Any, Set, Optional

import pandas as pd
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import Stream

from ADLClassifier import ADLClassifier, global_grace_period, grace_period_per_layer, extend_classifier_for_evaluation, \
    winning_layer_training, vectorized_for_loop, BaseLearningRateProgression, disabeling_deleted_layers, \
    delete_deleted_layers, input_preprocessing, add_weight_correction_parameter_to_user_choices, EMISSION_RECORDER_NAME
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper.add_weight_correction_parameter_to_user_choices import \
    ADD_WEIGHT_CORRECTION_PARAMETER_NAME
from Evaluation import config_handling
from Evaluation.PlottingFunctions import __plot_and_save_result, __compare_all_of_one_run
from Evaluation._config import ADWIN_DELTA_STANDIN, MAX_INSTANCES_TEST, STREAM_STRINGS, STANDARD_CONFIG, CLASSIFIERS, \
    STANDARD_CONFIG_WITH_CO2, UNSTABLE_CONFIG, UNSTABLE_STRING_IDX, STABLE_STRING_IDX, STABLE_CONFIG, \
    HYPERPARAMETER_KEYS, HYPERPARAMETERS, AMOUNT_OF_STRINGS, RESULTS_DIR_PATH, PROJECT_FOLDER_PATH, \
    STABLE_CONFIG_WITH_CO2, UNSTABLE_CONFIG_WITH_CO2, STANDARD_RUN_ID
from Evaluation.config_handling import adl_run_data_from_config, config_to_learner, config_to_stream, \
    standardize_learner_name


def __get_run_id() -> int:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    if any(RESULTS_DIR_PATH.iterdir()):
        *_, elem = RESULTS_DIR_PATH.iterdir()
        name, running_nr = elem.stem.split("=")
        return int(running_nr) + 1
    else:
        return 1


def standardize_hyperparamter_folder_name(old_name: str) -> str:
    """
    Standardizes the name of the hyperparameter folder
    :param old_name: old name of the hyperparameter folder
    :return: standardized name
    """
    name_parts = dict(tuple(p.split('=')) for p in old_name.split('_'))
    name_parts['classifier'] = standardize_learner_name(name_parts['classifier'])
    name_parts = list(name_parts.items())
    name_parts.sort(key=lambda x: x[0].lower())

    new_name = '_'.join(("=".join(part) for part in name_parts))
    if len(new_name) == 0 or len(dict(name_parts)['classifier']) == 0:
        raise Exception("i did something stupid again")

    return new_name


def directories_are_same(dir1, dir2):
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = dircmp(dir1, dir2)
    if (compared.left_only or compared.right_only or compared.diff_files
            or compared.funny_files):
        return False
    for subdir in compared.common_dirs:
        if not directories_are_same(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
            return False
    return True


def camel_case_split(str):
    """
    splits a string in camel case into word parts
    :param str: word in camel case
    :return: parts each beginning with a capital letter
    """
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word) for word in words]


def rename_folders(run_id: int):
    """Rename all folders under a given run id to fit the name of the hyperparameters used in the experiment."""
    logger = logging.getLogger('rename_folders')
    run_folder_path = RESULTS_DIR_PATH / f"runID={run_id}"
    if not run_folder_path.exists():
        logger.error(f'run folder {run_id} does not exist')
        raise Exception('run folder does not exist')

    i = 0
    tmp_folder = PROJECT_FOLDER_PATH / 'tmp' / f"runID={run_id}" / f'{i}'

    while True:
        if tmp_folder.exists() and directories_are_same(tmp_folder, run_folder_path):
            logger.info(f"runfolder already backed up under i={i}")
            break
        elif tmp_folder.exists():
            i += 1
            tmp_folder = PROJECT_FOLDER_PATH / 'tmp' / f"runID={run_id}" / f'{i}'
        else:
            logger.info(f"back up to i={i}")
            shutil.copytree(RESULTS_DIR_PATH / f"runID={run_id}", tmp_folder)
            break

    for hyperparameter_folder in [d for d in (RESULTS_DIR_PATH / f"runID={run_id}").iterdir() if d.is_dir()]:
        old_name = hyperparameter_folder.name
        new_name = standardize_hyperparamter_folder_name(old_name)
        shutil.move(RESULTS_DIR_PATH / f"runID={run_id}" / old_name, RESULTS_DIR_PATH / f"runID={run_id}" / new_name)


def _find_path_by_config_with_learner_object(run_id: int, config: Dict[str, Any], stream_name: str) -> Path:

    adl_parameter, rename_values, added_names = adl_run_data_from_config(
        config=config,
        with_weight_lr=(ADD_WEIGHT_CORRECTION_PARAMETER_NAME in config['learner'].name()),
        with_co2=(EMISSION_RECORDER_NAME in config['learner'].name()),
        learner_name=config['learner'].name()
    )
    hyperparameter_part_of_name_string = standardize_hyperparamter_folder_name("_".join((f"{str(key).replace('_', ' ')}={f'{value:.3g}' if type(value) == float else str(value).replace('_', ' ')}" for key, value in rename_values.items())))
    results_path = RESULTS_DIR_PATH / f"runID={run_id}" / hyperparameter_part_of_name_string / stream_name

    if not results_path.exists():
        config_description = hyperparameter_part_of_name_string.replace('_', '\n')
        logging.getLogger(f"logger_runID={run_id}").exception(f"no folder or result file exists for:\n config:\n {config_description},\n stream name: {stream_name},\n run id: {run_id},\n path:{results_path}\n\n")
        raise Exception(f'no folder exists for: {config}')

    return results_path


def __evaluate_on_stream(
        stream_data: Stream,
        run_id: int,
        adl_parameters: Dict[str, Any],
        classifier: type(ADLClassifier),
        rename_values: Dict[str, float],
        stream_name: Optional[str] = None,
        force: bool = False,
        run_name: Optional[str] = None
) -> None:
    logger = logging.getLogger(f"evaluate_on_stream_{run_name}:")
    call_identifier = run_name if not run_name is None else f'runID={run_id}'

    name_string_of_stream_data = stream_name if stream_name is not None else f"{stream_data.schema.dataset_name}/"
    hyperparameter_part_of_name_string = standardize_hyperparamter_folder_name("_".join((f"{str(key).replace('_', ' ')}={f'{value:.3g}' if type(value) == float else str(value).replace('_', ' ')}" for key, value in rename_values.items())))
    results_path = RESULTS_DIR_PATH / f"runID={run_id}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    if (results_path / "metrics.pickle").exists() and not force:
        logger.info(f"skipping {classifier.name()} + {name_string_of_stream_data} + {adl_parameters}, already evaluated, set force=True to overwrite")
        return

    logger.info(f"---------------Start time {call_identifier}: {datetime.now()}---------------")

    results_path.mkdir(parents=True, exist_ok=True)

    os.environ["CODECARBON_OUTPUT_DIR"] = results_path.absolute().as_posix()
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_LOG_LEVEL"] = "CRITICAL"

    logger.info(f"\nsummary for training {call_identifier}:\nrunId={run_id}\nstream={name_string_of_stream_data}\n" + "\n".join((f"{str(key).replace('_', ' ')}={str(value).replace('_', ' ')}" for key, value in rename_values.items())) + "\n----------")

    adl_classifier = classifier(
        schema=stream_data.schema,
        **adl_parameters
    )

    assert hasattr(adl_classifier, "record_of_model_shape"), f"ADL classifier {adl_classifier} does not keep track of model shape, and cannot be evaluated"

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES_TEST)
    total_time_end = time.time_ns()

    logger.info(f"total time spend training the network {call_identifier}: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")

    logger.info(f"\n\tThe cumulative results {call_identifier}:")
    logger.info(f"instances={results_ht.cumulative.metrics_dict()['instances']}, accuracy={results_ht.cumulative.metrics_dict()['accuracy']}")

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    # add custom parameters to result dict
    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end.insert(loc=0, column=key, value=[adl_classifier.record_of_model_shape[key][-1]])
        windowed_results[key] = adl_classifier.record_of_model_shape[key]
    metrics_at_end.insert(loc=0, column="overall time", value=((total_time_end-total_time_start) / 1e9))

    for key, val in rename_values.items():
        metrics_at_end.insert(loc=0, column=key, value=[str(val)])

    if results_path.parent.parent.name != f"runID={run_id}":
        print(results_path)
        raise FileNotFoundError
    metrics_at_end.to_pickle(results_path / "metrics.pickle")
    windowed_results.to_pickle(results_path / "metrics_per_window.pickle")
    adl_classifier.save_emissions_data((results_path / "emissions.csv").absolute().as_posix())

    # results_ht.write_to_file(results_path.absolute().as_posix())
    logger.info(f"---------------End time {call_identifier}: {datetime.now()}-----------------------")


def __write_summary(run_id: int, user_added_hyperparameter: Set[str]) -> None:
    runs_folder = RESULTS_DIR_PATH / f"runID={run_id}"
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
            runIdStr, hyperparameter_string, stream_name = Path(root).relative_to(PROJECT_FOLDER_PATH).as_posix().split("/")[2:]
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
    run_id = __get_run_id()
    logging.basicConfig(filename=Path(f"best_combination_runID={run_id}.log").absolute().as_posix(), level=logging.INFO)
    logger = logging.getLogger(f"logger_runID={run_id}")
    if stream_names is not None:
        assert len(stream_names) == len(streams), "give me as many names as streams or give me none"
        streams = list(zip(streams, stream_names))
    else:
        streams = list(zip(streams, [None] * len(streams)))
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

                                logger.info(f"---------------------------test: {current_run_index}/{total_nr_of_runs} = {current_run_index/total_nr_of_runs * 100 :.2f}%-----------------------------")
                                logger.info(f"------running since: {start}, for {(time.time_ns() - start_time) / (1e9 * 60) :.2f}min = {(time.time_ns() - start_time) / (1e9 * 60**2) :.2f}h------")
                                logger.info(f"---------------expected finish: {start + ((datetime.now() - start) * total_nr_of_runs/current_run_index)} ---------------")
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
                                logger.info(f"--------------------------test: {current_run_index}/{total_nr_of_runs} = {current_run_index/total_nr_of_runs * 100 :.2f}%-----------------------------")
                                logger.info(f"------running since: {start}, for {(time.time_ns() - start_time) / (1e9 * 60) :.2f}min = {(time.time_ns() - start_time) / (1e9 * 60**2) :.2f}h------")
                                logger.info(f"---------------expected finish: {start + ((datetime.now() - start) * total_nr_of_runs/current_run_index)} ---------------")
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

    if with_co_2:
        standard_config = STANDARD_CONFIG_WITH_CO2
    else:
        standard_config = STANDARD_CONFIG
    best_config = [standard_config.copy() for _ in range(nr_of_combinations)]

    # ADLWithInputProcessingWithoutForLoopWithWinningLayerTrainingWithGraphWithEmWithGlobalGracePeriodOf256Instances
    # Start time: 2025-03-17 17:14:04
    # End time: 2025-03-19 21:02:31

    classifiers = CLASSIFIERS
    run_id = STANDARD_RUN_ID
    logging.basicConfig(filename=Path(f"best_combination_runID={run_id}.log").absolute().as_posix(), level=logging.INFO)
    logger = logging.getLogger(f"logger_runID={run_id}")

    for i in range(nr_of_combinations):
        for classifier in classifiers:
            current_config = best_config[i]
            current_classifier = config_to_learner(*classifier, grace_period=(current_config['grace_period'], current_config['grace_type']), with_co2=with_co_2)
            logger.info(current_classifier.name())
            adl_parameter, rename_values, added_names = adl_run_data_from_config(current_config, with_weight_lr=('decoupled_lrs' in classifier), with_co2=with_co_2, learner_name=current_classifier.name())
            __evaluate_on_stream(
                classifier=current_classifier,
                stream_data=streams[i],
                stream_name=stream_names[i],
                adl_parameters=adl_parameter,
                rename_values=rename_values,
                run_id=run_id
            )
            __write_summary(run_id, added_names)
    __plot_and_save_result(run_id, show=False)


def _test_one_feature(stream_idx: int, classifier_idx: int, with_co_2: bool, run_name: str, force: bool = False) -> None:
    run_id = STANDARD_RUN_ID
    logging.basicConfig(filename=Path(f"best_combination_runID={run_id}.log").absolute().as_posix(), level=logging.INFO)
    logger = logging.getLogger(f"logger_runID={run_id}")
    current_config = STANDARD_CONFIG.copy()
    current_classifier = config_to_learner(*CLASSIFIERS[classifier_idx], grace_period=(current_config['grace_period'], current_config['grace_type']), with_co2=with_co_2)
    current_config['learner'] = current_classifier
    adl_parameter, rename_values, added_names = adl_run_data_from_config(current_config, with_weight_lr=('decoupled_lrs' in CLASSIFIERS[classifier_idx]), with_co2=with_co_2, learner_name=current_classifier.name())
    __evaluate_on_stream(
        classifier=current_classifier,
        stream_data=config_to_stream(STREAM_STRINGS[stream_idx]),
        stream_name=STREAM_STRINGS[stream_idx],
        adl_parameters=adl_parameter,
        rename_values=rename_values,
        run_id=run_id,
        run_name=run_name,
        force=force
    )
    __write_summary(run_id, added_names)


def _test_one_hyperparameter(hyperparameter_key_idx: int, hyperparameter_idx: int, stream_idx: int, with_co_2: bool, run_name: str, force: bool = False) -> None:
    assert 0 <= hyperparameter_key_idx < len(HYPERPARAMETER_KEYS), f"Invalid hyperparameter key index {hyperparameter_key_idx}"
    assert 0 <= hyperparameter_idx < len(HYPERPARAMETERS[HYPERPARAMETER_KEYS[hyperparameter_key_idx]]), f"invalid hyperparameter index {hyperparameter_idx}"
    assert 0 <= stream_idx < AMOUNT_OF_STRINGS, f"Invalid stream index {stream_idx}"
    run_id = STANDARD_RUN_ID
    if with_co_2:
        current_config = STANDARD_CONFIG_WITH_CO2.copy()
    else:
        current_config = STANDARD_CONFIG.copy()

    current_classifier = current_config['learner']
    hyperparameter_key = HYPERPARAMETER_KEYS[hyperparameter_key_idx]
    if hyperparameter_key == 'grace':
        current_config['grace_type'] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx][0]
        current_config['grace_period'] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx][1]
    else:
        current_config[hyperparameter_key] = HYPERPARAMETERS[hyperparameter_key][hyperparameter_idx]

    adl_parameter, rename_values, added_names = adl_run_data_from_config(current_config, with_weight_lr=True, with_co2=with_co_2, learner_name=current_config['learner'].name())
    __evaluate_on_stream(
        classifier=current_classifier,
        stream_data=config_to_stream(STREAM_STRINGS[stream_idx]),
        stream_name=STREAM_STRINGS[stream_idx],
        adl_parameters=adl_parameter,
        rename_values=rename_values,
        run_id=run_id,
        run_name=run_name,
        force=force
    )
    __write_summary(run_id, added_names)


def _test_stable(with_co_2: bool, run_name: str, force: bool = False) -> None:
    run_id = STANDARD_RUN_ID
    if with_co_2:
        current_config = STABLE_CONFIG_WITH_CO2.copy()
    else:
        current_config = STABLE_CONFIG.copy()
    current_classifier = current_config['learner']
    stream_idx = STABLE_STRING_IDX
    adl_parameter, rename_values, added_names = adl_run_data_from_config(current_config, with_weight_lr=True, with_co2=with_co_2, learner_name=current_config['learner'].name())
    __evaluate_on_stream(
        classifier=current_classifier,
        stream_data=config_to_stream(STREAM_STRINGS[stream_idx]),
        stream_name=STREAM_STRINGS[stream_idx],
        adl_parameters=adl_parameter,
        rename_values=rename_values,
        run_id=run_id,
        run_name=run_name,
        force=force
    )
    __write_summary(run_id, added_names)


def _test_unstable(with_co_2: bool, run_name: str, force: bool = False) -> None:
    run_id = STANDARD_RUN_ID
    if with_co_2:
        current_config = UNSTABLE_CONFIG_WITH_CO2.copy()
    else:
        current_config = UNSTABLE_CONFIG.copy()

    current_classifier = current_config['learner']
    stream_idx = UNSTABLE_STRING_IDX
    adl_parameter, rename_values, added_names = adl_run_data_from_config(current_config, with_weight_lr=True, with_co2=with_co_2, learner_name=current_config['learner'].name())
    __evaluate_on_stream(
        classifier=current_classifier,
        stream_data=config_to_stream(STREAM_STRINGS[stream_idx]),
        stream_name=STREAM_STRINGS[stream_idx],
        adl_parameters=adl_parameter,
        rename_values=rename_values,
        run_id=run_id,
        run_name=run_name,
        force=force
    )
    __write_summary(run_id, added_names)


def find_incomplete_directories(run_id: int) -> List[Path]:
    """finds the dirs of all incomplete directories in given run id result folder and raises if it found any"""
    run_folder_path = RESULTS_DIR_PATH / f"runID={run_id}"
    __write_summary(STANDARD_RUN_ID, )
    allowed_names = {'emissions.csv', 'metrics.pickle', 'metrics_per_window.pickle'}
    found = []
    for temp in run_folder_path.rglob('*'):
        if temp.is_dir() and len(list(temp.iterdir())) == 0:
            found.append(temp)
            continue
        elif temp.is_dir() and temp.parent.name != run_folder_path.name:
            if temp.name not in STREAM_STRINGS:
                found.append(temp)
        elif temp.is_file() and temp.name.startswith('summary'):
            continue
        elif temp.is_file() and (temp.name.startswith('emissions') or temp.name.startswith('metrics')):
            set_dir = set(file.name for file in temp.parent.iterdir())
            if any(name not in set_dir for name in allowed_names):
                found.append(temp.parent)
                continue
    logger = logging.getLogger('find_incomplete_directories')
    for found_path in set(found):
        logger.info(f'found incomplete dir: {found_path.name}\n{found_path.absolute()}')
    logger.info(f"found {len(found)} incomplete directories")
    return found


def clean_incomplete_directories(run_id: int) -> None:
    """removes the dirs of all incomplete directories in given run id result folder and raises if it found any"""
    logger = logging.getLogger('clean_incomplete_directories')
    found = find_incomplete_directories(run_id)

    for found_path in set(found):
        logger.info(f'removed incomplete dir: {found_path.name}\n{found_path.absolute()}')
        shutil.rmtree(found_path)
    logger.info(f"removed {len(found)} incomplete directories")