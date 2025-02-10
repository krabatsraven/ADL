from typing import List

from capymoa.drift.detectors import ADWIN
from capymoa.stream import ARFFStream
from ray import train

from ADLClassifier import extend_classifier_for_evaluation
from ADLClassifier.BaseClassifier import ADLClassifier
from Evaluation.RayTuneResources._config import MAX_INSTANCES
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import vectorized_for_loop, winning_layer_training, grace_period_per_layer, global_grace_period


def ADLTrainable(config):
    stream = ARFFStream(config['stream'])
    learner = config_to_learner(*config['learner'])

    if config['grace_period'] is not None and config['grace_period'][1] == "globel_grace":
        learner = grace_period_per_layer(config['grace_period'][0])(learner)
    elif config['grace_period'] is not None and config['grace_period'][1] == "layer_grace":
        learner = global_grace_period(config['grace_period'][0])(learner)

    learner = learner(
        schema=stream.get_schema(),
        lr=config['lr'],
        drift_detector=ADWIN(delta=config['adwin-delta']),
        mci_threshold_for_layer_pruning=config['mci'],
        loss_fn=config['loss_fn']
    )

    max_instances = MAX_INSTANCES
    nr_of_instances_seen = 0

    stream.restart()
    while stream.has_more_instances() and nr_of_instances_seen < max_instances:
        instance = stream.next_instance()
        learner.train(instance)
        nr_of_instances_seen += 1
        train.report({"score": learner.evaluator.accuracy(), 'instances_seen': nr_of_instances_seen})


def config_to_learner(*traits: str) -> type(ADLClassifier):
    decorators = []
    for trait in traits:
        match trait:
            case 'vectorized':
                decorators.append(vectorized_for_loop)
            case 'winning_layer':
                decorators.append(winning_layer_training)
            case _:
                raise ValueError(f"unknown trait: {trait}")

    return extend_classifier_for_evaluation(*decorators)
