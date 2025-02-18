from pathlib import Path
from typing import Optional

import pandas as pd
from ray import tune

from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNScheduler import SimpleDNNScheduler
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNSearchSpace import SimpleDNNSearchSpace
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNTrainable import SimpleDNNTrainable
from Evaluation.RayTuneResources.config_handling import write_config, load_config, evaluate_simple_dnn_config


def hyperparameter_search_for_SimpleDNN(
        nr_of_trials: int = 100,
        stream_name: str = 'electricity',
        nr_of_hidden_layers: int = 5,
        nr_of_neurons: int = 2**12,
        run_id: Optional[int] = None
):
    tuner = tune.Tuner(
        trainable=SimpleDNNTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=SimpleDNNScheduler,
        ),
        param_space=SimpleDNNSearchSpace(
            stream_name=stream_name, 
            nr_of_hidden_layers=nr_of_hidden_layers, 
            nr_of_neurons=nr_of_neurons
        ),
    )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    if run_id is not None:
        return write_config(best_result_config, run_id=run_id, run_name='ComparisonToADL')
    else:
        return write_config(best_result_config)


def compare_simple_to_adl(run_id, path_to_summary: Path, nr_of_trials: int):
    assert path_to_summary.exists(), "Path does not exist"
    assert path_to_summary.with_suffix('.csv')
    summary = pd.read_csv(path_to_summary, sep="\t")
    assert 'amount of nodes' in summary.columns.tolist(), "summary misses the nr of nodes in adl"
    assert 'amount of hidden layers' in summary.columns.tolist(), "summary misses the nr of hidden layers"

    hyperparameter_search_for_SimpleDNN(
        nr_of_trials=nr_of_trials,
        stream_name=summary.loc[:,'stream'].iloc[0],
        nr_of_neurons=int(summary.loc[:,'amount of nodes'].iloc[0]),
        nr_of_hidden_layers=int(summary.loc[:,'amount of hidden layers'].iloc[0]),
        run_id=run_id
    )
    config = load_config(run_id, run_name='ComparisonToADL')
    print(config)
    return evaluate_simple_dnn_config(config, run_id)

def compare_simple_to_adl_wo_evaluation(run_id, path_to_summary: Path, nr_of_trials: int):
    assert path_to_summary.exists(), "Path does not exist"
    assert path_to_summary.with_suffix('.csv')
    summary = pd.read_csv(path_to_summary, sep="\t")
    assert 'amount of nodes' in summary.columns.tolist(), "summary misses the nr of nodes in adl"
    assert 'amount of hidden layers' in summary.columns.tolist(), "summary misses the nr of hidden layers"

    hyperparameter_search_for_SimpleDNN(
        nr_of_trials=nr_of_trials,
        stream_name=summary.loc[:,'stream'].iloc[0],
        nr_of_neurons=int(summary.loc[:,'amount of nodes'].iloc[0]),
        nr_of_hidden_layers=int(summary.loc[:,'amount of hidden layers'].iloc[0]),
        run_id=run_id
    )
    return run_id

def evaluate_comparision_to_adl(run_id):
    config = load_config(run_id, run_name='ComparisonToADL')
    print(config)
    return evaluate_simple_dnn_config(config, run_id)