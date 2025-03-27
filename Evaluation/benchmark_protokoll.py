import asyncio
import logging
from pathlib import Path

from Evaluation.RayTuneResources import hyperparameter_search_for_ADL
from Evaluation.RayTuneResources.evaluate_runs import evaluate_adl_run
from Evaluation.EvaluationFunctions import _test_best_combination, _test_one_feature, _test_one_hyperparameter, \
    _test_stable, _test_unstable
from Evaluation._config import NR_OF_TRIALS, STREAM_STRINGS, AMOUNT_OF_CLASSIFIERS, AMOUNT_OF_STRINGS, \
    AMOUNT_HYPERPARAMETER_TESTS, AMOUNT_HYPERPARAMETERS_BEFORE, TOTAL_AMOUNT_HYPERPARAMETERS


def run_bench():
    # find best hyperparameters for all streams with adl
    #  stream                 lr     ...ght_learning_rate     adwin-delta           mci   grace_period           loss_fn       iter     total time (s)     score
    # electricity   0.190064                 0.153329        0.000239152   3.50888e-07   (128, 'layer_grace')   NLLLoss      45312           771.649    92.0418
    logging.basicConfig(filename=Path("benchmark.log").absolute().as_posix(), level=logging.INFO)
    print("hyperparameter for adl")
    runs = []
    # for stream in STREAM_STRINGS:
    #     print("stream: ", stream)
    #     runs.append(hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name=stream))

    # evaluate hyperparameters
    for run in [*range(47, 56)]:
        evaluate_adl_run(run)

    # compare to simple dnn: set a size that averages the adl node count
    runs.clear()
    # # compare to small simple dnn
    # print("hyperparameter for dnn")
    # for stream in STREAM_STRINGS:
    #     print("stream: ", stream)
    #     runs.append(hyperparameter_search_for_SimpleDNN(nr_of_trials=NR_OF_TRIALS, stream_name=stream))
    # runs = [*range(34, 43)]
    # for run in runs:
    #     evaluate_simple_run(run)

    # # run: best hyperparameter set also with co2 emission
    # # and run: best hyperparameter set for different classifier also with co2 emmisions:
    print("best combination")
    _test_best_combination(name="best_hyper_parameter_all_models_all_streams_with_co2", with_co_2=True)

    print("hyperparameter for adl")
    runs = [*range(47, 56)]
    for run in runs:
        stream = STREAM_STRINGS[run - 47]
        print("stream: ", stream)
        runs.append(hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name=stream, append_existing_run=run))

    for run in runs:
        evaluate_adl_run(run, force=True)


def bench_one_feature(run_idx: int, run_name: str) -> None:
    assert run_idx <= AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS - 1, f"run_idx {run_idx} out of range for feature test {AMOUNT_OF_STRINGS*AMOUNT_OF_CLASSIFIERS - 1}"
    stream_idx = run_idx // AMOUNT_OF_CLASSIFIERS
    classifier_idx = run_idx % AMOUNT_OF_CLASSIFIERS
    _test_one_feature(stream_idx=stream_idx, classifier_idx=classifier_idx, with_co_2=True, run_name=run_name)


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
    logging.basicConfig(filename=Path("bench_async.log").absolute().as_posix(), level=logging.INFO)
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