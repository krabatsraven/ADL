import logging
from pathlib import Path
from typing import Optional

import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import Tuner
from ray.tune.search.hyperopt import HyperOptSearch

from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainableUnstable, ADLTrainableStable
from Evaluation.RayTuneResources.ADL.UnstableSeachSpace import UnStableStableSearchSpace
from Evaluation.config_handling import write_config


def search_for_unstable_hyperparameters(run_id: int, nr_of_trials: int = 100, append_existing_run: Optional[int] = None):
    project_folder = Path(__file__).resolve().parent.parent.parent.parent.absolute()
    tmp_dir = project_folder / 'rayTmp'
    storage_path = tmp_dir / "results"
    logging.basicConfig(filename=(project_folder / 'hyper_parameter_search.log').as_posix(), level=logging.INFO)
    experiment_name = f"unstable_config_search_runID={run_id}"
    logger = logging.getLogger('search_for_unstable')
    ray.init(_temp_dir=tmp_dir.as_posix(), configure_logging=True, logging_level=logging.INFO)
    if append_existing_run is not None:
        run_id = append_existing_run

    if append_existing_run is not None:
        assert Path(f'results/runs/runID={append_existing_run}/config.json').exists(), "missing run config of run to append"
        predecessor_run_temp_files = (storage_path / experiment_name.removesuffix(f'_appending{append_existing_run}')).absolute()
        assert predecessor_run_temp_files.exists(), 'missing config of run to aapend'
        search_alg = HyperOptSearch(metric='score', mode='max').restore_from_dir(predecessor_run_temp_files.as_posix())
    else:
        search_alg = HyperOptSearch(metric='score', mode='max')
    if (storage_path / experiment_name).exists():
        tuner = Tuner.restore(
            (storage_path / experiment_name).as_posix(), trainable=ADLTrainableUnstable, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable=ADLTrainableUnstable,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                num_samples=nr_of_trials,
                scheduler=ADLScheduler,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=storage_path.as_posix(),
            ),
            param_space=UnStableStableSearchSpace,
        )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    out = write_config(best_result_config, run_id=run_id)
    ray.shutdown()
    return out


def search_for_stable_hyperparameters(run_id: int, stream_name: str, nr_of_trials: int = 100, append_existing_run: Optional[int] = None):
    project_folder = Path(__file__).resolve().parent.parent.parent.parent.absolute()
    tmp_dir = project_folder / 'rayTmp'
    storage_path = tmp_dir / "results"
    logging.basicConfig(filename=(project_folder / 'hyper_parameter_search.log').as_posix(), level=logging.INFO)
    experiment_name = f"stable_config_search_runID={run_id}"
    logger = logging.getLogger('search_for_stable')
    ray.init(_temp_dir=tmp_dir.as_posix(), configure_logging=True, logging_level=logging.INFO)
    if append_existing_run is not None:
        run_id = append_existing_run

    if append_existing_run is not None:
        assert Path(f'results/runs/runID={append_existing_run}/config.json').exists(), "missing run config of run to append"
        predecessor_run_temp_files = (storage_path / experiment_name.removesuffix(f'_appending{append_existing_run}')).absolute()
        assert predecessor_run_temp_files.exists(), 'missing config of run to append'
        search_alg = HyperOptSearch(metric='score', mode='max').restore_from_dir(predecessor_run_temp_files.as_posix())
    else:
        search_alg = HyperOptSearch(metric='score', mode='max')
    if (storage_path / experiment_name).exists():
        tuner = Tuner.restore(
            (storage_path / experiment_name).as_posix(), trainable=ADLTrainableStable, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable=ADLTrainableStable,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                num_samples=nr_of_trials,
                scheduler=ADLScheduler,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=storage_path.as_posix(),
            ),
            param_space=ADLSearchSpace(stream_name=stream_name, learner=('input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs')),
        )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    out = write_config(best_result_config, run_id=run_id)
    ray.shutdown()
    return out