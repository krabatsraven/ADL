from typing import Optional, Tuple

from capymoa.drift.detectors import ADWIN
from capymoa.stream import ARFFStream, Stream
from ray import train

from ADLClassifier import extend_classifier_for_evaluation
from ADLClassifier.BaseClassifier import ADLClassifier
from Evaluation.SynteticStreams.SynteticStreams import simple_agraval_single_drift, simple_agraval_three_drifts, simple_agraval_drift_back_and_forth
from Evaluation.EvaluationFunctions import __write_summary, __get_run_id, __evaluate_on_stream, __plot_and_save_result
from Evaluation._config import MAX_INSTANCES, ADWIN_DELTA_STANDIN
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


def evaluate_adl_run_config(config):
    classifier = config_to_learner(*config['learner'], grace_period=config['grace_period'])
    added_params = {
        "mci_threshold_for_layer_pruning": config['mci'],
        'drift_detector': ADWIN(config['adwin-delta']),
        'lr': config['lr'],
        'loss_fn': config['loss_fn'],

    }
    renames = {
        "MCICutOff": f"{config['mci']:4e}",
        ADWIN_DELTA_STANDIN: f"{config['adwin-delta']:.4e}",
        'classifier': config_to_learner(*config['learner'], grace_period=None).name(),
        'lr': f"{config['lr']:.4e}",
        'loss_fn': config['loss_fn'].__name__ if hasattr(config['loss_fn'], '__name__') else config['loss_fn'].__str__(),
    }
    added_names = {'MCICutOff', 'classifier', 'stream', ADWIN_DELTA_STANDIN, 'lr', 'loss_fn'}

    if config['grace_period'] is not None and config['grace_period'][1] == 'global_grace':
        renames['globalGracePeriod'] = config['grace_period'][0]
        added_names.add('globalGracePeriod')

    elif config['grace_period'] is not None and config['grace_period'][1] == 'layer_grace':
        renames['gracePeriodPerLayer'] = config['grace_period'][0]
        added_names.add('gracePeriodPerLayer')

    run_id = __get_run_id()
    __evaluate_on_stream(
        stream_data=config_to_stream(config['stream']),
        run_id=run_id,
        classifier=classifier,
        adl_parameters=added_params,
        rename_values=renames,
        stream_name=config['stream'],
    )
    __write_summary(run_id, added_names)
    __plot_and_save_result(run_id, show=False)
