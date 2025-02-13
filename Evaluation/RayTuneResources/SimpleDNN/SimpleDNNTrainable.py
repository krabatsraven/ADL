from capymoa.evaluation import ClassificationEvaluator
from ray import train

from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier
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
