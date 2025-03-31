import asyncio
import datetime
import logging
from pathlib import Path

from Evaluation.RayTuneResources import hyperparameter_search_for_ADL, hyperparameter_search_for_SimpleDNN
from Evaluation.RayTuneResources.ADL.search_for_unstable_hyperparameters import search_for_unstable_hyperparameters, \
    search_for_stable_hyperparameters
from Evaluation.RayTuneResources.evaluate_runs import evaluate_adl_run, evaluate_simple_run
from Evaluation.EvaluationFunctions import _test_best_combination, _test_one_feature, _test_one_hyperparameter, \
    _test_stable, _test_unstable
from Evaluation._config import NR_OF_TRIALS, STREAM_STRINGS, AMOUNT_OF_CLASSIFIERS, AMOUNT_OF_STRINGS, \
    AMOUNT_HYPERPARAMETER_TESTS, AMOUNT_HYPERPARAMETERS_BEFORE, TOTAL_AMOUNT_HYPERPARAMETERS, UNSTABLE_STRING_IDX


def run_bench():
    '''benchmark programs'''
    skip_ray_adl_hyperparameter = True
    skip_ray_stable_unstable = False
    skip_eval_adl_hyperparameter = True
    skip_ray_dnn_hyperparamter = True
    skip_eval_dnn_hyperparameter = True
    skip_best_combination = True
    skip_single_function = False
    skip_single_hyperparameter = False
    # todo: replace stable and unstable config from finding above
    skip_stable = True
    skip_unstable = True
    skip_ray_adl_hyperparameter_append = False
    skip_eval_adl_hyperparameter_append = False


    logging.basicConfig(filename=Path("benchmark.log").absolute().as_posix(), level=logging.INFO)
    logger = logging.getLogger('run_bench')
    runs = []

    if not skip_ray_adl_hyperparameter:
        # find the best hyperparameters for all streams with adl
        logger.info(f"starting hyperparameter search for adl: {datetime.datetime.now()}")
        for stream in STREAM_STRINGS:
            logger.info(f"stream: {stream}")
            runs.append(hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name=stream))
        return
    else:
        logger.info(f'skipping hyperparameter search for adl')

    if not skip_ray_stable_unstable:
        logger.info(f'starting search for unstable: {datetime.datetime.now()}')
        search_for_unstable_hyperparameters(run_id=999, nr_of_trials=2000)
        # todo: update unstable string idx with found stream idx from unstable search
        return 
        logger.info(f'starting search for stable: {datetime.datetime.now()}')
        search_for_stable_hyperparameters(stream_name=STREAM_STRINGS[UNSTABLE_STRING_IDX], run_id=998, nr_of_trials=2000)
        return
    else:
        logger.info('skipping hyperparameter search for unstable/stable adl')


    low_run_id = 47
    high_run_id_border = low_run_id + 9
    if not skip_eval_adl_hyperparameter:
        logger.info(f'starting evaluation of hyperparameter runs {low_run_id} - {high_run_id_border - 1}: {datetime.datetime.now()}')
        # evaluate hyperparameters
        for run in range(low_run_id, high_run_id_border):
            evaluate_adl_run(run)
    else:
        logger.info(f'skipping evaluation of hyperparameter runs {low_run_id} - {high_run_id_border - 1}')

    # compare to simple dnn: set a size that averages the adl node count
    runs.clear()
    # # compare to small simple dnn
    if not skip_ray_dnn_hyperparamter:
        logger.info(f"starting hyperparameter search for dnn: {datetime.datetime.now()}")
        for stream in STREAM_STRINGS:
            logger.info(f"stream: {stream}")
            runs.append(hyperparameter_search_for_SimpleDNN(nr_of_trials=NR_OF_TRIALS, stream_name=stream))
    else:
        logger.info(f"skipping hyperparameter search for dnn")

    if not skip_eval_dnn_hyperparameter:
        logger.info(f"starting evaluating dnn: {datetime.datetime.now()}")
        runs = [*range(34, 43)]
        for run in runs:
            evaluate_simple_run(run)
    else:
        logger.info(f"skipping evaluating dnn")

    if not skip_best_combination:
        # run: best hyperparameter set also with co2 emission
        # and run: best hyperparameter set for different classifier also with co2 emmisions:
        logger.info(f"starting best combination benching: {datetime.datetime.now()}")
        _test_best_combination(name="best_hyper_parameter_all_models_all_streams_with_co2", with_co_2=True)
    else:
        logger.info(f"skipping best combination benching")

    run_idx = 0
    biggest_run_index = AMOUNT_HYPERPARAMETER_TESTS + (AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS) + 2 - 1
    if not skip_single_function:
        logger.info(f'starting single function benching: {datetime.datetime.now()}')
        for stream_idx in range(AMOUNT_OF_STRINGS):
            for classifier_idx in range(AMOUNT_OF_CLASSIFIERS):
                bench_one_feature(run_idx, f'{run_idx}/{biggest_run_index}')
                run_idx += 1
    else:
        logger.info(f'skipping single function benching')
        for stream_idx in range(AMOUNT_OF_STRINGS):
            for classifier_idx in range(AMOUNT_OF_CLASSIFIERS):
                run_idx += 1

    if not skip_single_hyperparameter:
        logger.info(f'starting single hyperparameter benching: {datetime.datetime.now()}')
        for _ in range(AMOUNT_HYPERPARAMETER_TESTS):
            bench_one_hyperparameter_isolated(run_idx, f'{run_idx}/{biggest_run_index}' )
            run_idx += 1
    else:
        logger.info(f'skipping single hyperparameter benching')
        for _ in range(AMOUNT_HYPERPARAMETER_TESTS):
            run_idx += 1

    if not skip_stable:
        logger.info(f'starting stable benching: {datetime.datetime.now()}')
        bench_stable(run_idx, f'{run_idx}/{biggest_run_index}')
    else:
        logger.info('skipping stable benching')
    run_idx += 1

    if not skip_unstable:
        logger.info(f'starting unstable benching: {datetime.datetime.now()}')
        bench_unstable(run_idx, f'{run_idx}/{biggest_run_index}')
    else:
        logger.info('skipping unstable benching')

    if not skip_ray_adl_hyperparameter_append:
        logger.info(f'starting hyperparameter search for adl again: {datetime.datetime.now()}')
        runs = [*range(low_run_id, high_run_id_border)]
        for run in runs:
            stream = STREAM_STRINGS[run - low_run_id]
            print("stream: ", stream)
            runs.append(hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name=stream, append_existing_run=run))
            return
    else:
        logger.info(f'skipping hyperparameter search for adl again')

    if not skip_eval_adl_hyperparameter_append:
        logger.info(f'starting evaluation of appended hyperparameter: {datetime.datetime.now()}')
        for run in runs:
            evaluate_adl_run(run, force=True)
    else:
        logger.info('skipping evaluation of appended hyperparameter')


def bench_one_feature(run_idx: int, run_name: str) -> None:
    assert run_idx <= AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS - 1, f"run_idx {run_idx} out of range for feature test {AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS - 1}"
    stream_idx = run_idx // AMOUNT_OF_CLASSIFIERS
    classifier_idx = run_idx % AMOUNT_OF_CLASSIFIERS
    _test_one_feature(
        stream_idx=stream_idx,
        classifier_idx=classifier_idx,
        with_co_2=True, 
        run_name=run_name
    )


def bench_one_hyperparameter_isolated(run_idx: int, run_name: str) -> None:
    relevant_run_idx = run_idx - (AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS)
    assert 0 <= relevant_run_idx <= AMOUNT_HYPERPARAMETER_TESTS, f"run_idx: {run_idx}, relevant: {relevant_run_idx} out of range for hyperparameter test {AMOUNT_HYPERPARAMETER_TESTS}"
    key_idx_plus_idx = relevant_run_idx % TOTAL_AMOUNT_HYPERPARAMETERS
    tmp = [amount_before > key_idx_plus_idx for amount_before in AMOUNT_HYPERPARAMETERS_BEFORE]
    hyperparameter_key_idx = tmp.index(True) - 1 if True in tmp else len(AMOUNT_HYPERPARAMETERS_BEFORE) - 1
    hyperparameter_idx = key_idx_plus_idx - AMOUNT_HYPERPARAMETERS_BEFORE[hyperparameter_key_idx]
    stream_idx = relevant_run_idx // TOTAL_AMOUNT_HYPERPARAMETERS
    _test_one_hyperparameter(
        hyperparameter_key_idx=hyperparameter_key_idx,
        hyperparameter_idx=hyperparameter_idx,
        stream_idx=stream_idx,
        with_co_2=True,
        run_name=run_name
    )


def bench_stable(run_idx: int, run_name: str) -> None:
    relevant_run_idx = run_idx - (AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS - 1) - AMOUNT_HYPERPARAMETER_TESTS
    assert relevant_run_idx == 1, f"run_idx: {run_idx}, relevant: {relevant_run_idx} out of range for stable test {1}"
    _test_stable(with_co_2=True, run_name=run_name)


def bench_unstable(run_idx: int, run_name: str) -> None:
    relevant_run_idx = run_idx - (AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS - 1) - AMOUNT_HYPERPARAMETER_TESTS
    assert relevant_run_idx == 2, f"run_idx: {run_idx}, relevant: {relevant_run_idx} out of range for stable test {2}"
    _test_unstable(with_co_2=True, run_name=run_name)


async def async_run_feature_test(run_idx: int, run_name:str) -> None:
    await asyncio.to_thread(bench_one_feature, run_idx, run_name)


async def async_run_hyper_test(run_idx: int, run_name:str) -> None:
    await asyncio.to_thread(bench_one_hyperparameter_isolated,run_idx, run_name)


async def async_run_stable(run_idx: int, run_name: str) -> None:
    await asyncio.to_thread(bench_stable, run_idx, run_name)


async def async_run_unstable(run_idx: int, run_name:str) -> None:
    await asyncio.to_thread(bench_unstable, run_idx, run_name)


async def bench_async():
    # logging.basicConfig(filename=Path("bench_async.log").absolute().as_posix(), level=logging.INFO)
    tasks = []

    biggest_run_index = AMOUNT_HYPERPARAMETER_TESTS + (AMOUNT_OF_STRINGS * AMOUNT_OF_CLASSIFIERS) + 2 - 1
    run_idx = 0
    for stream_idx in range(AMOUNT_OF_STRINGS):
        for classifier_idx in range(AMOUNT_OF_CLASSIFIERS):
            tasks.append(async_run_feature_test(run_idx, f'{run_idx}/{biggest_run_index}'))
            run_idx += 1

    for _ in range(AMOUNT_HYPERPARAMETER_TESTS):
        tasks.append(async_run_hyper_test(run_idx, f'{run_idx}/{biggest_run_index}'))
        run_idx += 1

    tasks.append(async_run_stable(run_idx, f'{run_idx}/{biggest_run_index}'))
    run_idx += 1

    tasks.append(async_run_unstable(run_idx, f'{run_idx}/{biggest_run_index}'))

    await asyncio.gather(*tasks)