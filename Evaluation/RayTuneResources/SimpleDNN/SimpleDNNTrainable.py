import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from capymoa.evaluation import ClassificationEvaluator, prequential_evaluation
from ray import train

from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier
from Evaluation.EvaluationFunctions import __get_run_id
from Evaluation.RayTuneResources.ADL.ADLTrainable import config_to_stream
from Evaluation._config import MAX_INSTANCES, WINDOW_SIZE


def SimpleDNNTrainable(config):  # ①
    stream = config_to_stream(config['stream'])
    learner = SimpleDNNClassifier(
        schema=stream.get_schema(),
        model_structure=config['model_structure'],
        lr=config["lr"]
    )
    max_instances = MAX_INSTANCES
    nr_of_instances_seen = 0
    window_size = WINDOW_SIZE

    stream.restart()

    evaluator = ClassificationEvaluator(stream.schema, window_size=window_size)
    while stream.has_more_instances() and nr_of_instances_seen < max_instances:
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        learner.train(instance)
        evaluator.update(instance.y_index, prediction)
        nr_of_instances_seen += 1
        train.report({"score": evaluator.accuracy(), 'instances_seen': nr_of_instances_seen})

def evaluate_simple_dnn_config(config):
    stream = config_to_stream(config['stream'])
    learner = SimpleDNNClassifier(
        schema=stream.schema,
        lr=config['lr'],
        model_structure=config['model_structure'],
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
    results_path = Path("results/runs") / f"runID={run_id}" / f"lr={config['lr']:.4e}_modelStructure={config['model_structure']}" / config['stream']
    results_path.mkdir(parents=True, exist_ok=True)
    results_at_end = pd.DataFrame([results_ht.cumulative.metrics()], columns=results_ht.cumulative.metrics_header())
    results_at_end['lr'] = config['lr']
    results_at_end.insert(loc=0, column='model_structure', value=str(config['model_structure']))
    results_at_end.to_pickle(results_path / "metrics.pickle")
    results_ht.windowed.metrics_per_window().to_pickle(results_path / "metrics_per_window.pickle")
    results_at_end.to_csv(results_path / "summary.csv")
