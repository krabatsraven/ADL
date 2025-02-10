from ray import tune

from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNScheduler import SimpleDNNScheduler
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNSearchSpace import SimpleDNNSearchSpace
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNTrainable import SimpleDNNTrainable


def hyperparameter_search_for_SimpleDNN(nr_of_trials: int = 100):
    tuner = tune.Tuner(
        trainable=SimpleDNNTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=SimpleDNNScheduler,
        ),
        param_space=SimpleDNNSearchSpace
    )

    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="max").config)
