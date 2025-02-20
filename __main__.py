import pathlib

import ray

from Evaluation.EvaluationFunctions import _test_example

from Evaluation import evaluate_adl_run, evaluate_simple_run, hyperparameter_search_for_SimpleDNN, \
    hyperparameter_search_for_ADL, MAX_INSTANCES, __compare_results_via_plot_and_save
from Evaluation.RayTuneResources.SimpleDNN.hyperparameter_search_for_SimpleDNN import compare_simple_to_adl, evaluate_comparision_to_adl, get_simple_arguments
from Evaluation.RayTuneResources.config_handling import load_config, evaluate_adl_run_config
from Evaluation._config import NR_OF_TRIALS

# 82% acc: {'learner': ('vectorized', 'winning_layer', 'decoupled_lrs'), 'stream': 'electricity', 'lr': 0.17037433308206834, 'layer_weight_learning_rate': 0.0051048969488651065, 'adwin-delta': 2.2019797256079463e-05, 'mci': 2.3105218391180886e-07, 'grace_period': (32, 'global_grace'), 'loss_fn': 'NLLLoss'}

if __name__ == "__main__":
    # _test_example()

    stream_strings = [
        'electricity',
        'agraval_no_drift', 'agraval_single_drift', 'agraval_three_drifts', 'agraval_drift_back_and_forth',
        'sea_no_drift', 'sea_single_drift', 'sea_three_drifts', 'sea_drift_back_and_forth'
    ]
    runs = []

    print("running adl with higher min instances")
    # todo: run mit min_run=4000 für electricity again
    runs.clear()
    run_id_2 = hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name='electricity')
    evaluate_adl_run(run_id_2)
    tasks_2 = get_simple_arguments(run_id=run_id_2, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run_id_2}/summary.csv')), nr_of_trials=NR_OF_TRIALS)
    for task in tasks_2:
        runs.append(hyperparameter_search_for_SimpleDNN(**task))
    evaluate_comparision_to_adl(run_id_2)

    print("comparing to adl runs 3 - 7")
    # todo: finish runs
    runs.clear()
    for run in range(3,8):
        tasks = get_simple_arguments(run_id=run, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run}/summary.csv')), nr_of_trials=NR_OF_TRIALS)
        for task in tasks:
            runs.append(hyperparameter_search_for_SimpleDNN(**task))
    print("evaluating the simple runs")
    for run in runs:
        evaluate_comparision_to_adl(run)

    print("comparing to adl runs 9-10")
    runs.clear()
    for run in range(9,11):
        tasks = get_simple_arguments(run_id=run, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run}/summary.csv')), nr_of_trials=NR_OF_TRIALS)
        for task in tasks:
            runs.append(hyperparameter_search_for_SimpleDNN(**task))
    print("evaluating the simple runs")
    for run in runs:
        evaluate_comparision_to_adl(run)

    # todo: fix that multiple comparisions overwrite each other
    # print("comparing to adl runs 8")
    # runs.clear()
    # tasks = get_simple_arguments(run_id=8, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run}/summary.csv')), nr_of_trials=NR_OF_TRIALS)
    # for task in tasks:
    #     runs.append(hyperparameter_search_for_SimpleDNN(**task))
    # print("evaluating the simple runs")
    # for run in runs:
    #     evaluate_comparision_to_adl(run)