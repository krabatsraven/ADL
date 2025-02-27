import logging
import tempfile
from pathlib import Path
from typing import Tuple

import ray
from grpc.experimental import experimental_api
from ray import tune, train
from ray.train import RunConfig
from ray.tune import Tuner
from ray.tune.search.hyperopt import HyperOptSearch

from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainable
from Evaluation.config_handling import write_config


def hyperparameter_search_for_ADL(nr_of_trials: int = 100, stream_name: str = 'electricity', learner: Tuple[str, ...] = ('input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs')):
    print("started adl hyperparameter search for ", stream_name)
    tmp_dir = Path("~/rayTmp").absolute().resolve()
    ray.init(_temp_dir=tmp_dir.as_posix(), configure_logging=True, logging_level=logging.INFO)

    storage_path = tmp_dir / "results"
    experiment_name = f"ADL_{learner}_{stream_name}"

    search_alg = HyperOptSearch(metric='score', mode='max')
    run_config = RunConfig(
        name=experiment_name,
        storage_path=storage_path.as_posix(),
    )

    if (storage_path / experiment_name).exists():
        print("storage path exists: restoring from storage path")
        # search_alg.restore_from_dir((storage_path / experiment_name).as_posix())
        tuner = Tuner.restore(
            (storage_path / experiment_name).as_posix(), trainable=ADLTrainable, resume_errored=True)
        # tuner = tune.Tuner(
        #     trainable=ADLTrainable,
        #     tune_config=tune.TuneConfig(
        #         search_alg=search_alg,
        #         num_samples=nr_of_trials,
        #         scheduler=ADLScheduler,
        #     ),
        #     run_config=run_config
        # )
    else:
        tuner = tune.Tuner(
            trainable=ADLTrainable,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                num_samples=nr_of_trials,
                scheduler=ADLScheduler,
            ),
            run_config=run_config,
            param_space=ADLSearchSpace(stream_name=stream_name, learner=learner),
        )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    out = write_config(best_result_config)
    ray.shutdown()
    return out
