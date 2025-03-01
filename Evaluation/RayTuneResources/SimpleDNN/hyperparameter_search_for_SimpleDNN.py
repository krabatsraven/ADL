import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import Tuner
from ray.tune.search.hyperopt import HyperOptSearch

from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNScheduler import SimpleDNNScheduler
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNSearchSpace import SimpleDNNSearchSpace
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNTrainable import SimpleDNNTrainable
from Evaluation.RayTuneResources.evaluate_runs import evaluate_simple_dnn_config
from Evaluation.config_handling import write_config, load_config


def hyperparameter_search_for_SimpleDNN(
        nr_of_trials: int = 100,
        stream_name: str = 'electricity',
        nr_of_hidden_layers: int = 5,
        nr_of_neurons: int = 2**12,
        run_id: Optional[int] = None
):

    # more than 10 hidden layers are too much
    if nr_of_hidden_layers >= 9:
        nr_of_hidden_layers = 8
    if nr_of_neurons > 2**14:
        nr_of_neurons = 2**14

    print("started simple dnn hyperparameter search for ", stream_name)
    tmp_dir = Path("./rayTmp").absolute().resolve()
    ray.init(_temp_dir=tmp_dir.as_posix(), configure_logging=True, logging_level=logging.INFO)

    storage_path = tmp_dir / "results"
    experiment_name = f"SimpleDNN_{nr_of_hidden_layers}layers_{nr_of_neurons}neurons_{stream_name}_{stream_name}"

    if (storage_path / experiment_name).exists():
        tuner = Tuner.restore(
            (storage_path / experiment_name).as_posix(), trainable=SimpleDNNTrainable, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable=SimpleDNNTrainable,
            tune_config=tune.TuneConfig(
                num_samples=nr_of_trials,
                search_alg=HyperOptSearch(metric='score', mode='max'),
                scheduler=SimpleDNNScheduler,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=storage_path.as_posix(),
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
        out = write_config(best_result_config, run_id=run_id, run_name='ComparisonToADL')
    else:
        out = write_config(best_result_config)
    ray.shutdown()
    return out


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

def get_simple_arguments(run_id, path_to_summary: Path, nr_of_trials: int):
    assert path_to_summary.exists(), "Path does not exist"
    assert path_to_summary.with_suffix('.csv')
    summary = pd.read_csv(path_to_summary, sep="\t")
    assert 'amount of nodes' in summary.columns.tolist(), "summary misses the nr of nodes in adl"
    assert 'amount of hidden layers' in summary.columns.tolist(), "summary misses the nr of hidden layers"

    return [{'nr_of_trials': nr_of_trials, 'stream_name': summary.loc[:,'stream'].iloc[trial], 'nr_of_neurons': int(summary.loc[:,'amount of nodes'].iloc[trial]), 'nr_of_hidden_layers': int(summary.loc[:,'amount of hidden layers'].iloc[trial]), 'run_id': run_id} for trial in range(len(summary.index))]

def evaluate_comparision_to_adl(run_id):
    config = load_config(run_id, run_name='ComparisonToADL')
    print(config)
    return evaluate_simple_dnn_config(config, run_id)