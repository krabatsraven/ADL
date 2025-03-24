import asyncio
import logging
from pathlib import Path

from Evaluation.RayTuneResources import hyperparameter_search_for_ADL, hyperparameter_search_for_SimpleDNN
from Evaluation.RayTuneResources.evaluate_runs import evaluate_adl_run, evaluate_simple_run
from Evaluation.EvaluationFunctions import _test_best_combination, _test_one_combination
from Evaluation._config import NR_OF_TRIALS, STREAM_STRINGS, CLASSIFIERS


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


def run_bench_mogon(stream_idx: int, classifier_idx: int) -> None:
    run_idx = stream_idx * len(CLASSIFIERS) + classifier_idx
    if run_idx == 0:
        logger = logging.getLogger(f"logger_runID={99}")
        logger.info("Starting MOGON RUN")
    logging.basicConfig(filename=Path("mogon_run.log").absolute().as_posix(), level=logging.INFO)
    run_name = f"{run_idx}/{len(STREAM_STRINGS)*len(CLASSIFIERS) - 1}"
    _test_one_combination(stream_idx=stream_idx, classifier_idx=classifier_idx, with_co_2=True, run_name=run_name)
    if run_idx == len(STREAM_STRINGS)*len(CLASSIFIERS):
        logger = logging.getLogger(f"logger_runID={99}")
        logger.info("FINISHED MOGON RUN")


async def async_run_bench(stream_idx: int, classifier_idx: int) -> None:
    await asyncio.to_thread(run_bench_mogon, stream_idx, classifier_idx)


async def bench_async():
    tasks = []
    for stream_idx in range(len(STREAM_STRINGS)):
        for classifier_idx in range(len(CLASSIFIERS)):
            tasks.append(async_run_bench(stream_idx, classifier_idx))

    await asyncio.gather(*tasks)