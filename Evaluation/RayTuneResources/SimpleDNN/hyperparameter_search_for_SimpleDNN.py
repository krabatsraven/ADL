import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from capymoa.evaluation import prequential_evaluation
from ray import tune

from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier
from Evaluation.EvaluationFunctions import __get_run_id, __write_summary
from Evaluation.RayTuneResources._config import MAX_INSTANCES
from Evaluation.RayTuneResources.ADL.ADLTrainable import config_to_stream
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNScheduler import SimpleDNNScheduler
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNSearchSpace import SimpleDNNSearchSpace
from Evaluation.RayTuneResources.SimpleDNN.SimpleDNNTrainable import SimpleDNNTrainable


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

    stream = config_to_stream(best_result_config['stream'])
    learner = SimpleDNNClassifier(
        schema=stream.schema,
        lr=best_result_config['lr'],
        model_structure=best_result_config['model_structure'],
    )
    print("--------------------------------------------------------------------------")
    print(f"---------------Start time: {datetime.now()}---------------------")
    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream, learner=learner, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES)
    total_time_end = time.time_ns()
    print(f"---------------End time: {datetime.now()}-----------------------")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"instances={results_ht.cumulative.metrics_dict()['instances']}, accuracy={results_ht.cumulative.metrics_dict()['accuracy']}")
    print("--------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------")

    run_id = __get_run_id()
    results_path = Path("results/runs") / f"runID={run_id}" / f"lr={best_result_config['lr']:.4e}_modelStructure={best_result_config['model_structure']}" / best_result_config['stream']
    results_path.mkdir(parents=True, exist_ok=True)
    results_at_end = pd.DataFrame([results_ht.cumulative.metrics()], columns=results_ht.cumulative.metrics_header())
    results_at_end['lr'] = best_result_config['lr']
    results_at_end.insert(loc=0, column='model_structure', value=str(best_result_config['model_structure']))
    results_at_end.to_pickle(results_path / "metrics.pickle")
    results_ht.windowed.metrics_per_window().to_pickle(results_path / "metrics_per_window.pickle")
    results_at_end.to_csv(results_path / "summary.csv")
