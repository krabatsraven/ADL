import os
import tempfile

import torch
from capymoa.evaluation import ClassificationEvaluator
from ray import train
from ray.train import Checkpoint

from Evaluation.config_handling import config_to_stream
from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier
from Evaluation._config import MAX_INSTANCES, WINDOW_SIZE, MIN_INSTANCES


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

    checkpoint = train.get_checkpoint()
    start = 0
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=False)
            start = checkpoint_dict['instances_seen'] + 1
            learner.state_dict = checkpoint_dict['learner']
            assert learner.learning_rate == config['lr']

    nr_of_instances_seen = 0

    stream.restart()
    while stream.has_more_instances and nr_of_instances_seen < start and nr_of_instances_seen < MAX_INSTANCES:
        # fast-forward to start
        _ = stream.next_instance()
        nr_of_instances_seen += 1

    metrics = {"score": 0, 'instances_seen': nr_of_instances_seen}
    evaluator = ClassificationEvaluator(stream.schema, window_size=window_size)
    while stream.has_more_instances() and nr_of_instances_seen < max_instances:
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        learner.train(instance)
        evaluator.update(instance.y_index, prediction)
        nr_of_instances_seen += 1
        if nr_of_instances_seen % max(MIN_INSTANCES // 100, 1) == 0:
            metrics = {"score": evaluator.accuracy(), 'instances_seen': nr_of_instances_seen}
            if nr_of_instances_seen % max(MIN_INSTANCES, 1) == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    torch.save(
                        {'instances_seen': nr_of_instances_seen, 'learner': learner.state_dict},
                        os.path.join(tmpdir, 'checkpoint.pt')
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))
            else:
                train.report(metrics=metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {'instances_seen': nr_of_instances_seen, 'learner': learner.state_dict},
            os.path.join(tmpdir, 'checkpoint.pt')
        )
        train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))
