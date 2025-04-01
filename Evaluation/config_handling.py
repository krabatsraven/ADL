import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Set

import pandas as pd
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Stream, ARFFStream
from torch.nn import CrossEntropyLoss

import Evaluation
from ADLClassifier import grace_period_per_layer, extend_classifier_for_evaluation, \
    winning_layer_training, vectorized_for_loop, ADLClassifier, add_weight_correction_parameter_to_user_choices, \
    input_preprocessing, disabeling_deleted_layers, delete_deleted_layers
from ADLClassifier import NETWORK_GRAPH_NAME, EMISSION_RECORDER_NAME, DELETE_DELETED_LAYERS_NAME, \
    DISABLED_DELETED_LAYERS_NAME, WINNING_LAYER_TRAINING_NAME, VECTORIZED_FOR_LOOP_NAME, INPUT_PREPROCESSING_NAME, \
    ADD_WEIGHT_CORRECTION_PARAMETER_NAME, GRACE_PERIOD_PER_LAYER_NAME, GLOBAL_GRACE_PERIOD_NAME
from ADLClassifier.Resources.NLLLoss import NLLLoss

from Evaluation.SynteticStreams.SynteticAgrawalStreams import agrawal_no_drift, agrawal_single_drift, agrawal_three_drifts, agrawal_four_drifts
from Evaluation.SynteticStreams.SyntheticSEAStreams import sea_no_drift, sea_single_drift, sea_three_drifts, sea_four_drifts
from Evaluation._config import ADWIN_DELTA_STANDIN, LEARNER_PART_NAMES


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


def adl_run_data_from_config(config, with_weight_lr: bool, with_co2: bool = False, learner_name: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    added_params = {
        "mci_threshold_for_layer_pruning": config['mci'],
        'drift_detector': ADWIN(config['adwin-delta']),
        'lr': config['lr'],
        'loss_fn': config_to_loss_fn(config['loss_fn']),
    }
    renames = {
        "MCICutOff": f"{config['mci']:4e}",
        ADWIN_DELTA_STANDIN: f"{config['adwin-delta']:.4e}",
        'classifier': standardize_learner_name(config_to_learner(*config['learner'], grace_period=None, with_co2=with_co2).name() if learner_name is None else learner_name),
        'lr': f"{config['lr']:.4e}",
        'loss_fn': config['loss_fn'],
    }
    added_names = {'MCICutOff', 'classifier', 'stream', ADWIN_DELTA_STANDIN, 'lr', 'loss_fn'}

    if config['grace_period'] is not None and config['grace_type'] == 'global_grace':
        renames['globalGracePeriod'] = config['grace_period']
        added_names.add('globalGracePeriod')
    elif config['grace_period'] is not None and config['grace_type'] == 'layer_grace':
        renames['gracePeriodPerLayer'] = config['grace_period']
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
            return ARFFStream(Path('data/electricity.arff').absolute().as_posix())
        case 'electricity_tiny':
            return ARFFStream(Path('data/electricity_tiny.arff').absolute().as_posix())
        case 'agraval_no_drift':
            return agrawal_no_drift
        case 'agraval_single_drift':
            return agrawal_single_drift
        case 'agraval_three_drifts':
            return agrawal_three_drifts
        case 'agraval_four_drifts':
            return agrawal_four_drifts
        case 'sea_no_drift':
            return sea_no_drift
        case 'sea_single_drift':
            return sea_single_drift
        case 'sea_three_drifts':
            return sea_three_drifts
        case 'sea_four_drifts':
            return sea_four_drifts
        case _:
            raise ValueError(f"unknown stream: {stream_name}")


def config_to_learner(*traits: str, grace_period: Optional[Tuple[int, str]], with_co2: bool = False) -> type(ADLClassifier):
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

    if grace_period is not None and grace_period[1] == "global_grace":
        decorators.append(grace_period_per_layer(grace_period[0]))
    elif grace_period is not None and grace_period[1] == "layer_grace":
        decorators.append(grace_period_per_layer(grace_period[0]))

    learner = extend_classifier_for_evaluation(*decorators, with_emissions=with_co2)

    return learner


def config_to_loss_fn(loss_fn_name: str):
    match loss_fn_name:
        case 'CrossEntropyLoss':
            return CrossEntropyLoss()
        case 'NLLLoss':
            return NLLLoss


def standardize_learner_name(learner_name: str) -> str:
    parts = []

    if NETWORK_GRAPH_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[NETWORK_GRAPH_NAME])
    elif LEARNER_PART_NAMES[NETWORK_GRAPH_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[NETWORK_GRAPH_NAME])
    elif 'WithGraph' in learner_name:
        parts.append(LEARNER_PART_NAMES[NETWORK_GRAPH_NAME])

    if EMISSION_RECORDER_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[EMISSION_RECORDER_NAME])
    elif LEARNER_PART_NAMES[EMISSION_RECORDER_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[EMISSION_RECORDER_NAME])
    elif 'WithEmissions' in learner_name:
        parts.append(LEARNER_PART_NAMES[EMISSION_RECORDER_NAME])

    if DISABLED_DELETED_LAYERS_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[DISABLED_DELETED_LAYERS_NAME])
    elif LEARNER_PART_NAMES[DISABLED_DELETED_LAYERS_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[DISABLED_DELETED_LAYERS_NAME])
    elif DELETE_DELETED_LAYERS_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[DELETE_DELETED_LAYERS_NAME])
    elif LEARNER_PART_NAMES[DELETE_DELETED_LAYERS_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[DELETE_DELETED_LAYERS_NAME])

    if WINNING_LAYER_TRAINING_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[WINNING_LAYER_TRAINING_NAME])
    elif LEARNER_PART_NAMES[WINNING_LAYER_TRAINING_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[WINNING_LAYER_TRAINING_NAME])
    elif 'WithWinning' in learner_name:
        parts.append(LEARNER_PART_NAMES[WINNING_LAYER_TRAINING_NAME])

    if VECTORIZED_FOR_LOOP_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[VECTORIZED_FOR_LOOP_NAME])
    elif LEARNER_PART_NAMES[VECTORIZED_FOR_LOOP_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[VECTORIZED_FOR_LOOP_NAME])
    elif 'WithoutForLoop' in learner_name:
        parts.append(LEARNER_PART_NAMES[VECTORIZED_FOR_LOOP_NAME])

    if INPUT_PREPROCESSING_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[INPUT_PREPROCESSING_NAME])
    elif LEARNER_PART_NAMES[INPUT_PREPROCESSING_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[INPUT_PREPROCESSING_NAME])
    elif 'WithInputProcessing' in learner_name:
        parts.append(LEARNER_PART_NAMES[INPUT_PREPROCESSING_NAME])

    if ADD_WEIGHT_CORRECTION_PARAMETER_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[ADD_WEIGHT_CORRECTION_PARAMETER_NAME])
    elif LEARNER_PART_NAMES[ADD_WEIGHT_CORRECTION_PARAMETER_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[ADD_WEIGHT_CORRECTION_PARAMETER_NAME])
    elif 'WithUserChosenWeight' in learner_name:
        parts.append(LEARNER_PART_NAMES[ADD_WEIGHT_CORRECTION_PARAMETER_NAME])

    if GRACE_PERIOD_PER_LAYER_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[GRACE_PERIOD_PER_LAYER_NAME])
    elif LEARNER_PART_NAMES[GRACE_PERIOD_PER_LAYER_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[GRACE_PERIOD_PER_LAYER_NAME])
    elif 'WithGracePeriodPerLayer' in learner_name:
        parts.append(LEARNER_PART_NAMES[GRACE_PERIOD_PER_LAYER_NAME])

    if GLOBAL_GRACE_PERIOD_NAME in learner_name:
        parts.append(LEARNER_PART_NAMES[GLOBAL_GRACE_PERIOD_NAME])
    elif LEARNER_PART_NAMES[GLOBAL_GRACE_PERIOD_NAME] in learner_name:
        parts.append(LEARNER_PART_NAMES[GLOBAL_GRACE_PERIOD_NAME])
    elif 'WithGlobalGrace' in learner_name:
        parts.append(LEARNER_PART_NAMES[GLOBAL_GRACE_PERIOD_NAME])

    parts = sorted(list(set(parts)))
    return ''.join(parts)