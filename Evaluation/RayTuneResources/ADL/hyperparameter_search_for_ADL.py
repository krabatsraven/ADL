import logging
from typing import Tuple

import ray
from ray import tune

from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainable
from Evaluation.RayTuneResources.config_handling import write_config


def hyperparameter_search_for_ADL(nr_of_trials: int = 100, stream_name: str = 'electricity', learner: Tuple[str, ...] = ('vectorized', 'winning_layer', 'decoupled_lrs')):
    print("started adl hyperparameter search for ", stream_name)
    ray.init(_temp_dir='/home/david/rayTmp', configure_logging=True, logging_level=logging.INFO)
    tuner = tune.Tuner(
        trainable=ADLTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=ADLScheduler,
        ),
        param_space=ADLSearchSpace(stream_name=stream_name, learner=learner),
    )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    out = write_config(best_result_config)
    ray.shutdown()
    return out
