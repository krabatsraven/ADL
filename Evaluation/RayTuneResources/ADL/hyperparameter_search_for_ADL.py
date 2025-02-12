from ray import tune


from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainable, evaluate_config


def hyperparameter_search_for_ADL(nr_of_trials: int = 100, stream_name: str = 'electricity'):
    tuner = tune.Tuner(
        trainable=ADLTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=ADLScheduler,
        ),
        param_space=ADLSearchSpace(stream_name=stream_name),
    )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    evaluate_config(best_result_config)
