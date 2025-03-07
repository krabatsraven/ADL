import logging
from pathlib import Path
from typing import Tuple, Optional

import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import Tuner
from ray.tune.search import Searcher
from ray.tune.search.hyperopt import HyperOptSearch

from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainable
from Evaluation.config_handling import write_config


def hyperparameter_search_for_ADL(nr_of_trials: int = 100, stream_name: str = 'electricity', learner: Tuple[str, ...] = ('input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs'), run_id: Optional[int] = None, append_existing_run: Optional[int] = None):
    if run_id is not None and append_existing_run is not None:
        assert(append_existing_run == run_id), "both need to be the same"
    if append_existing_run is not None and run_id is None:
        run_id = append_existing_run

    print("started adl hyperparameter search for ", stream_name)
    tmp_dir = Path("./rayTmp").absolute().resolve()
    ray.init(_temp_dir=tmp_dir.as_posix(), configure_logging=True, logging_level=logging.INFO)

    storage_path = tmp_dir / "results"
    experiment_name = f"ADL_{learner}_{stream_name}{'' if append_existing_run is None else f'_appending{append_existing_run}'}"

    if append_existing_run is not None:
        assert Path(f'results/runs/runID={append_existing_run}/config.json').exists(), "missing run config of run to append"
        predecessor_run_temp_files = (storage_path / experiment_name.removesuffix(f'_appending{append_existing_run}')).absolute()
        assert predecessor_run_temp_files.exists(), 'missing config of run to aapend'
        search_alg = Searcher().restore_from_dir(predecessor_run_temp_files.as_posix())
    else:
        search_alg = HyperOptSearch(metric='score', mode='max')

    if (storage_path / experiment_name).exists():
        tuner = Tuner.restore(
            (storage_path / experiment_name).as_posix(), trainable=ADLTrainable, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable=ADLTrainable,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                num_samples=nr_of_trials,
                scheduler=ADLScheduler,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=storage_path.as_posix(),
            ),
            param_space=ADLSearchSpace(stream_name=stream_name, learner=learner),
        )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    out = write_config(best_result_config, run_id=run_id)
    ray.shutdown()
    return out
