import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from capymoa.datasets import Electricity
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import Stream, ARFFStream
from torch.nn import CrossEntropyLoss

from ADLClassifier import global_grace_period, grace_period_per_layer, extend_classifier_for_evaluation, \
    winning_layer_training, vectorized_for_loop, ADLClassifier, add_weight_correction_parameter_to_user_choices, \
    input_preprocessing, disabeling_deleted_layers, delete_deleted_layers
from ADLClassifier.Resources.NLLLoss import NLLLoss
from Evaluation import agrawal_no_drift
from Evaluation.EvaluationFunctions import __plot_and_save_result
from Evaluation.SynteticStreams import agrawal_single_drift, agrawal_three_drifts, agrawal_drift_back_and_forth
from Evaluation.SynteticStreams.SyntheticSEAStreams import sea_no_drift, sea_single_drift, sea_three_drifts, \
    sea_drift_back_and_forth
from Evaluation._config import ADWIN_DELTA_STANDIN, MAX_INSTANCES, MAX_INSTANCES_TEST
from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier
from Evaluation.EvaluationFunctions import __get_run_id, __write_summary, __evaluate_on_stream


def write_config(config, run_id: Optional[int] = None, run_name: str = ''):
    if run_id is None:
        run_id = __get_run_id()
    else:
        run_id = run_id
    save_dir = Path("results/runs") / f"runID={run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # todo: if config exists, change config name?
    with open(save_dir / f"config{run_name}.json", "w") as f:
        f.write(json.dumps(config, indent=4))
    return run_id

def load_config(run_id: int, run_name: str= ''):
    save_file = Path("results/runs") / f"runID={run_id}" / f"config{run_name}.json"
    if not save_file.exists():
        raise FileNotFoundError("Config not found, aborting")
    else:
        with open(save_file, "r") as f:
            return json.loads(f.read())


def evaluate_adl_run(run_id):
    config = load_config(run_id)
    return evaluate_adl_run_config(config, run_id)


def evaluate_simple_run(run_id):
    config = load_config(run_id)
    return evaluate_simple_dnn_config(config, run_id)


def evaluate_simple_dnn_config(config, run_id):
    stream = config_to_stream(config['stream'])
    learner = SimpleDNNClassifier(
        schema=stream.schema,
        lr=config['lr'],
        model_structure=config['model_structure'],
    )
    print("--------------------------------------------------------------------------")
    print(f"---------------Start time: {datetime.now()}---------------------")
    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream, learner=learner, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES_TEST)
    total_time_end = time.time_ns()
    print(f"---------------End time: {datetime.now()}-----------------------")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"instances={results_ht.cumulative.metrics_dict()['instances']}, accuracy={results_ht.cumulative.metrics_dict()['accuracy']}")
    print("--------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------")

    results_path = Path("results/runs") / f"runID={run_id}" / f"lr={config['lr']:.4e}_modelStructure={config['model_structure']}" / config['stream']
    results_path.mkdir(parents=True, exist_ok=True)
    results_at_end = pd.DataFrame([results_ht.cumulative.metrics()], columns=results_ht.cumulative.metrics_header())
    results_at_end['lr'] = config['lr']
    results_at_end.insert(loc=0, column='model_structure', value=str(config['model_structure']))
    results_at_end.to_pickle(results_path / "metrics.pickle")
    results_ht.windowed.metrics_per_window().to_pickle(results_path / "metrics_per_window.pickle")
    results_at_end.to_csv(results_path / "summary.csv")


def evaluate_adl_run_config(config, run_id):
    classifier = config_to_learner(*config['learner'], grace_period=config['grace_period'])
    added_params = {
        "mci_threshold_for_layer_pruning": config['mci'],
        'drift_detector': ADWIN(config['adwin-delta']),
        'lr': config['lr'],
        'loss_fn': config_to_loss_fn(config['loss_fn']),
    }
    renames = {
        "MCICutOff": f"{config['mci']:4e}",
        ADWIN_DELTA_STANDIN: f"{config['adwin-delta']:.4e}",
        'classifier': config_to_learner(*config['learner'], grace_period=None).name(),
        'lr': f"{config['lr']:.4e}",
        'loss_fn': config['loss_fn'],
    }
    added_names = {'MCICutOff', 'classifier', 'stream', ADWIN_DELTA_STANDIN, 'lr', 'loss_fn'}

    if config['grace_period'] is not None and config['grace_period'][1] == 'global_grace':
        renames['globalGracePeriod'] = config['grace_period'][0]
        added_names.add('globalGracePeriod')
    elif config['grace_period'] is not None and config['grace_period'][1] == 'layer_grace':
        renames['gracePeriodPerLayer'] = config['grace_period'][0]
        added_names.add('gracePeriodPerLayer')
    if 'WithUserChosenWeightLR' in classifier.name():
        added_params['layer_weight_learning_rate'] = config['layer_weight_learning_rate']
        renames['layerWeightLR'] = config['layer_weight_learning_rate']
        added_names.add('layerWeightLR')

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


def config_to_stream(stream_name: str) -> type(Stream):
    match stream_name:
        case 'electricity':
            return ARFFStream(str(Path('data/electricity.arff').absolute().resolve()))
        case 'electricity_tiny':
            return ARFFStream(str(Path('data/electricity_tiny.arff').absolute().resolve()))
        case 'agraval_no_drift':
            return agrawal_no_drift
        case 'agraval_single_drift':
            return agrawal_single_drift
        case 'agraval_three_drifts':
            return agrawal_three_drifts
        case 'agraval_drift_back_and_forth':
            return agrawal_drift_back_and_forth
        case 'sea_no_drift':
            return sea_no_drift
        case 'sea_single_drift':
            return sea_single_drift
        case 'sea_three_drifts':
            return sea_three_drifts
        case 'sea_drift_back_and_forth':
            return sea_drift_back_and_forth
        case _:
            raise ValueError(f"unknown stream: {stream_name}")


def config_to_learner(*traits: str, grace_period: Optional[Tuple[int, str]]) -> type(ADLClassifier):
    decorators = []
    for trait in traits:
        match trait:
            case 'vectorized':
                decorators.append(vectorized_for_loop)
            case 'winning_layer':
                decorators.append(winning_layer_training)
            case 'decoupled_lrs':
                decorators.append(add_weight_correction_parameter_to_user_choices)
            case 'delete_deleted_layer':
                decorators.append(delete_deleted_layers)
            case 'disable_deleted_layer':
                decorators.append(disabeling_deleted_layers)
            case 'input_preprocessing':
                decorators.append(input_preprocessing)
            case _:
                raise ValueError(f"unknown trait: {trait}")

    learner = extend_classifier_for_evaluation(*decorators)

    if grace_period is not None and grace_period[1] == "global_grace":
        learner = grace_period_per_layer(grace_period[0])(learner)
    elif grace_period is not None and grace_period[1] == "layer_grace":
        learner = global_grace_period(grace_period[0])(learner)

    return learner


def config_to_loss_fn(loss_fn_name: str):
    match loss_fn_name:
        case 'CrossEntropyLoss':
            return CrossEntropyLoss()
        case 'NLLLoss':
            return NLLLoss