from ray import tune

from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainable


def hyperparameter_search_for_ADL(nr_of_trials: int = 100):
    tuner = tune.Tuner(
        trainable=ADLTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=ADLScheduler,
        ),
        param_space=ADLSearchSpace
    )

    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="max").config)