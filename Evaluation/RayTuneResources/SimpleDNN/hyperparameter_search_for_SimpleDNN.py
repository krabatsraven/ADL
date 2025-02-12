from ray import tune

from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNScheduler import SimpleDNNScheduler
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNSearchSpace import SimpleDNNSearchSpace
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNTrainable import SimpleDNNTrainable, evaluate_simple_dnn_config


def hyperparameter_search_for_SimpleDNN(nr_of_trials: int = 100, stream_name: str = 'electricity'):
    tuner = tune.Tuner(
        trainable=SimpleDNNTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=SimpleDNNScheduler,
        ),
        param_space=SimpleDNNSearchSpace(stream_name=stream_name),
    )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)
    evaluate_simple_dnn_config(best_result_config)
