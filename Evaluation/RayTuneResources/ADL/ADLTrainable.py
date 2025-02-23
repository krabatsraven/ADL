from capymoa.drift.detectors import ADWIN
from ray import train

from Evaluation.config_handling import config_to_stream, config_to_learner, config_to_loss_fn
from Evaluation._config import MAX_INSTANCES


def ADLTrainable(config):
    stream = config_to_stream(config['stream'])
    learner = config_to_learner(*config['learner'], grace_period=config['grace_period'])

    if 'WithUserChosenWeightLR' in learner.name():
        learner = learner(
            schema=stream.get_schema(),
            lr=config['lr'],
            drift_detector=ADWIN(delta=config['adwin-delta']),
            mci_threshold_for_layer_pruning=config['mci'],
            loss_fn=config_to_loss_fn(config['loss_fn']),
            layer_weight_learning_rate=config['layer_weight_learning_rate']
        )
    else:
        learner = learner(
            schema=stream.get_schema(),
            lr=config['lr'],
            drift_detector=ADWIN(delta=config['adwin-delta']),
            mci_threshold_for_layer_pruning=config['mci'],
            loss_fn=config_to_loss_fn(config['loss_fn'])
        )

    max_instances = MAX_INSTANCES
    nr_of_instances_seen = 0

    stream.restart()
    while stream.has_more_instances() and nr_of_instances_seen < max_instances:
        instance = stream.next_instance()
        learner.train(instance)
        nr_of_instances_seen += 1
        train.report({"score": learner.evaluator.accuracy(), 'instances_seen': nr_of_instances_seen})
