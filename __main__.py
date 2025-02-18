import pathlib

import ray

from Evaluation.EvaluationFunctions import _test_example

from Evaluation import evaluate_adl_run, evaluate_simple_run, hyperparameter_search_for_SimpleDNN, \
    hyperparameter_search_for_ADL, MAX_INSTANCES
from Evaluation.RayTuneResources.SimpleDNN.hyperparameter_search_for_SimpleDNN import compare_simple_to_adl, \
    compare_simple_to_adl_wo_evaluation, evaluate_comparision_to_adl
from Evaluation.RayTuneResources.config_handling import load_config, evaluate_adl_run_config
from Evaluation._config import NR_OF_TRIALS

# 82% acc: {'learner': ('vectorized', 'winning_layer', 'decoupled_lrs'), 'stream': 'electricity', 'lr': 0.17037433308206834, 'layer_weight_learning_rate': 0.0051048969488651065, 'adwin-delta': 2.2019797256079463e-05, 'mci': 2.3105218391180886e-07, 'grace_period': (32, 'global_grace'), 'loss_fn': 'NLLLoss'}

if __name__ == "__main__":
    # _test_example()

    ray.init(_temp_dir='/home/david/rayTmp')
    ray.shutdown()
    stream_strings = [
        'electricity',
        'agraval_no_drift', 'agraval_single_drift', 'agraval_three_drifts', 'agraval_drift_back_and_forth',
        'sea_no_drift', 'sea_single_drift', 'sea_three_drifts', 'sea_drift_back_and_forth'
    ]
    runs = []
    # for stream_name in stream_strings:
    #     runs.append(hyperparameter_search_for_ADL(NR_OF_TRIALS, stream_name=stream_name))
    for run in range(8,11):
        evaluate_adl_run(run)

    print("evaluating adl run 1 with different loss function")
    # todo: test loss functions against each other
    # just use config from run one and change loss function string
    # after that run compare with links
    config = load_config(1)
    config['loss_fn'] = 'CrossEntropyLoss'
    run_id = evaluate_adl_run_config(config, 1)

    print("comparing to adl runs")

    # todo: finish runs
    runs.clear()
    ray.init(_temp_dir='/home/david/rayTmp')
    for run in range(2,11):
        runs.append(compare_simple_to_adl_wo_evaluation(run, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run}/summary.csv')), nr_of_trials=NR_OF_TRIALS))
    ray.shutdown()
    print("evaluating the simple runs")
    for run in runs:
        evaluate_comparision_to_adl(run)

    # todo: run best LR Coupled
    runs.clear()
    ray.init(_temp_dir='/home/david/rayTmp')
    for stream_name in stream_strings:
        runs.append(hyperparameter_search_for_ADL(NR_OF_TRIALS, stream_name=stream_name, learner=('vectorized', 'winning_layer')))
    ray.shutdown()

    print("searching for simple hyperparameters with default values")
    ray.init(_temp_dir='/home/david/rayTmp')
    # todo: test simple with default
    for stream_name in stream_strings:
        hyperparameter_search_for_SimpleDNN(NR_OF_TRIALS, stream_name=stream_name)
    ray.shutdown()

    print("running adl with higher min instances")
    # todo: run mit min_run=4000 für electricity again
    MIN_INSTANCES=int(MAX_INSTANCES * 0.1)
    ray.init(_temp_dir='/home/david/rayTmp')
    run_id_2 = hyperparameter_search_for_ADL(NR_OF_TRIALS, stream_name='electricity')
    ray.shutdown()
    MIN_INSTANCES = int(MAX_INSTANCES * 0.01)
    evaluate_adl_run(run_id_2)
    ray.init(_temp_dir='/home/david/rayTmp')
    compare_simple_to_adl_wo_evaluation(run_id_2, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run_id}/summary.csv')), nr_of_trials=NR_OF_TRIALS)
    ray.shutdown()
    evaluate_comparision_to_adl(run_id_2)