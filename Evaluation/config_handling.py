import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Set

import pandas as pd
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Stream, ARFFStream
from torch.nn import CrossEntropyLoss

import Evaluation
from ADLClassifier import global_grace_period, grace_period_per_layer, extend_classifier_for_evaluation, \
    winning_layer_training, vectorized_for_loop, ADLClassifier, add_weight_correction_parameter_to_user_choices, \
    input_preprocessing, disabeling_deleted_layers, delete_deleted_layers
from ADLClassifier.Resources.NLLLoss import NLLLoss

from Evaluation.SynteticStreams import agrawal_no_drift, agrawal_single_drift, agrawal_three_drifts, agrawal_drift_back_and_forth
from Evaluation.SynteticStreams.SyntheticSEAStreams import sea_no_drift, sea_single_drift, sea_three_drifts, \
    sea_drift_back_and_forth
from Evaluation._config import ADWIN_DELTA_STANDIN


def get_best_config_for_stream_name(stream_name: str) -> Dict[str, Any]:
    run_folder = Path('results/runs').absolute().resolve()
    possible_configs = []
    for folder in run_folder.iterdir():
        if (folder / "config.json").exists() and (folder / 'summary.csv').exists():
            summary = pd.read_csv((folder / 'summary.csv'), sep='\t')
            for _, row in summary[summary['stream'] == stream_name].iterrows():
                possible_configs.append((row['accuracy'], (folder / "config.json")))

    assert len(possible_configs) > 0, "No configs found for stream"
    with open(max(possible_configs, key=lambda x: x[0])[1], "r") as f:
        return json.loads(f.read())


def adl_run_data_from_config(config, with_weight_lr: bool) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
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
    if with_weight_lr:
        added_params['layer_weight_learning_rate'] = config['layer_weight_learning_rate']
        renames['layerWeightLR'] = config['layer_weight_learning_rate']
        added_names.add('layerWeightLR')

    return added_params, renames, added_names



def write_config(config, run_id: Optional[int] = None, run_name: str = ''):
    if run_id is None:
        run_id = Evaluation.EvaluationFunctions.__get_run_id()
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