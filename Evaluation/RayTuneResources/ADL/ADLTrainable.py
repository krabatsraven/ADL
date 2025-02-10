from pathlib import Path
from typing import List, Optional, Tuple

from capymoa.drift.detectors import ADWIN
from capymoa.stream import ARFFStream, Stream
from ray import train

from ADLClassifier import extend_classifier_for_evaluation
from ADLClassifier.BaseClassifier import ADLClassifier
from Evaluation import simple_agraval_single_drift, simple_agraval_three_drifts, simple_agraval_drift_back_and_forth
from Evaluation.RayTuneResources._config import MAX_INSTANCES
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import vectorized_for_loop, winning_layer_training, grace_period_per_layer, global_grace_period


def ADLTrainable(config):
    stream = config_to_stream(config['stream'])
    learner = config_to_learner(*config['learner'], grace_period=config['grace_period'])

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


def config_to_learner(*traits: str, grace_period: Optional[Tuple[int, str]]) -> type(ADLClassifier):
    decorators = []
    for trait in traits:
        match trait:
            case 'vectorized':
                decorators.append(vectorized_for_loop)
            case 'winning_layer':
                decorators.append(winning_layer_training)
            case _:
                raise ValueError(f"unknown trait: {trait}")

    learner = extend_classifier_for_evaluation(*decorators)

    if grace_period is not None and grace_period[1] == "global_grace":
        learner = grace_period_per_layer(grace_period[0])(learner)
    elif grace_period is not None and grace_period[1] == "layer_grace":
        learner = global_grace_period(grace_period[0])(learner)

    return learner


def config_to_stream(stream_name: str) -> type(Stream):
    match stream_name:
        case 'electricity':
            return ARFFStream('/home/david/PycharmProjects/ADL/data/electricity.arff')
        case 'electricity_tiny':
            return ARFFStream('/home/david/PycharmProjects/ADL/data/electricity_tiny.arff')
        case 'simple_agraval_single_drift':
            return simple_agraval_single_drift
        case 'simple_agraval_three_drifts':
            return simple_agraval_three_drifts
        case 'simple_agraval_drift_back_and_forth':
            return simple_agraval_drift_back_and_forth
        case _:
            raise ValueError(f"unknown stream: {stream_name}")
