from Evaluation.RayTuneResources import hyperparameter_search_for_ADL, hyperparameter_search_for_SimpleDNN
from Evaluation.RayTuneResources.evaluate_runs import evaluate_adl_run, evaluate_simple_run
from Evaluation.EvaluationFunctions import _test_best_combination
from Evaluation._config import NR_OF_TRIALS, STREAM_STRINGS

def run_bench():
    # find best hyperparameters for all streams with adl
    print("hyperparameter for adl")
    runs = []
    for stream in STREAM_STRINGS:
        print("stream: ", stream)
        runs.append(hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name=stream))

    # evaluate hyperparameters
    for run in runs:
        evaluate_adl_run(run)

    # compare to simple dnn: set a size that averages the adl node count
    runs.clear()
    # compare to small simple dnn
    print("hyperparameter for dnn")
    for stream in STREAM_STRINGS:
        print("stream: ", stream)
        runs.append(hyperparameter_search_for_SimpleDNN(nr_of_trials=NR_OF_TRIALS, stream_name=stream))

    for run in runs:
        evaluate_simple_run(run)

    # run: best hyperparameter set also with co2 emission
    # and run: best hyperparameter set for different classifier also with co2 emmisions:
    _test_best_combination(name="best_hyper_parameter_all_models_all_streams_with_co2", with_co_2=True)
