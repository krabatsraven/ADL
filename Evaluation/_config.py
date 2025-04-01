from itertools import combinations
from pathlib import Path

from ADLClassifier import extend_classifier_for_evaluation, grace_period_per_layer, input_preprocessing, \
    vectorized_for_loop, winning_layer_training, add_weight_correction_parameter_to_user_choices, NETWORK_GRAPH_NAME, \
    EMISSION_RECORDER_NAME, DELETE_DELETED_LAYERS_NAME, DISABLED_DELETED_LAYERS_NAME, WINNING_LAYER_TRAINING_NAME, \
    VECTORIZED_FOR_LOOP_NAME, INPUT_PREPROCESSING_NAME, ADD_WEIGHT_CORRECTION_PARAMETER_NAME, \
    GRACE_PERIOD_PER_LAYER_NAME, GLOBAL_GRACE_PERIOD_NAME

PROJECT_FOLDER_PATH = Path(__file__).parent.parent.resolve().absolute()
RESULTS_DIR_PATH = PROJECT_FOLDER_PATH / 'results' / 'runs'
STANDARD_RUN_ID = 9999

LEARNER_PART_NAMES = {
    NETWORK_GRAPH_NAME: 'Graph',
    EMISSION_RECORDER_NAME: 'Emission',
    DELETE_DELETED_LAYERS_NAME: 'Delete',
    DISABLED_DELETED_LAYERS_NAME: 'Disable',
    WINNING_LAYER_TRAINING_NAME: 'Winning',
    VECTORIZED_FOR_LOOP_NAME: 'Vector',
    INPUT_PREPROCESSING_NAME: 'Input',
    ADD_WEIGHT_CORRECTION_PARAMETER_NAME: 'Weight',
    GRACE_PERIOD_PER_LAYER_NAME: 'Layer',
    GLOBAL_GRACE_PERIOD_NAME: 'Global',
}


LEARNER_CONFIG_TO_NAMES = {
    'input_preprocessing' : 'Input Normalization and One-Hot-Encoding',
    'vectorized': 'Vectorized MCI Calculation',
    'winning_layer': 'Winning Layer Training',
    'delete_deleted_layer': 'Delete Pruned Layer',
    'disable_deleted_layer': 'Disable Pruned Layer',
    'decoupled_lrs': 'Different Learning Rates for Optimizer and Voting Weights',
    'layer_grace': 'Grace Period Per Layer',
    'global_grace': 'Global Grace Period',
}

NR_OF_TRIALS = 500
MAX_INSTANCES = 100000
MAX_INSTANCES_TEST = MAX_INSTANCES
MIN_INSTANCES = MAX_INSTANCES // 20
WINDOW_SIZE = 100
ADWIN_DELTA_STANDIN = "adwin-delta"
CONCEPT_LENGTH = (MIN_INSTANCES // 5) * 4
MAX_RECURRENCES_PER_STREAM = MAX_INSTANCES_TEST // CONCEPT_LENGTH

STREAM_STRINGS = [
    'electricity',
    'agraval_no_drift', 'agraval_single_drift', 'agraval_three_drifts', 'agraval_four_drifts',
    'sea_no_drift', 'sea_single_drift', 'sea_three_drifts', 'sea_four_drifts'
]

STREAM_NAMES = {
    'electricity': 'Electricity',
    'agraval_no_drift': 'Agrawal Without Drift',
    'agraval_single_drift': 'Agrawal With One Drift',
    'agraval_three_drifts': 'Agrawal With Three Drifts',
    'agraval_four_drifts': 'Agrawal With Four Drifts',
    'sea_no_drift': 'SEA Without Drift',
    'sea_single_drift': 'SEA With One Drift',
    'sea_three_drifts': 'SEA With Three Drifts', 
    'sea_four_drifts': 'SEA With Four Drifts'
}

STANDARD_LEARNER = [grace_period_per_layer(369), add_weight_correction_parameter_to_user_choices, winning_layer_training, vectorized_for_loop, input_preprocessing]
STANDARD_LEARNER.reverse()

STANDARD_CONFIG = {
    'lr': 0.104083,
    'learner': extend_classifier_for_evaluation(*STANDARD_LEARNER),
    'layer_weight_learning_rate': 0.450039,
    'adwin-delta': 3.63053e-05,
    'mci': 9.34633e-07,
    'grace_period': 369,
    'grace_type': 'layer_grace',
    'loss_fn': 'NLLLoss',
}

STANDARD_CONFIG_WITH_CO2 = STANDARD_CONFIG.copy()
STANDARD_CONFIG_WITH_CO2['learner'] = extend_classifier_for_evaluation(*STANDARD_LEARNER, with_emissions=True)
stable_learner = [global_grace(495), add_weight_correction_parameter_to_user_choices, winning_layer_training, vectorized_for_loop, input_preprocessing]
STABLE_CONFIG = {
    {'learner': extend_classifier_for_evaluation(*stable_learner),
     'lr': 0.22970622156817536,
     'layer_weight_learning_rate': 0.24126254519960913,
     'adwin-delta': 1.6083049259877988e-07,
     'mci': 1.2538978365240957e-06,
     'grace_type': 'global_grace',
     'grace_period': 495,
     'loss_fn': 'NLLLoss'}
}
STABLE_CONFIG_WITH_CO2 = STABLE_CONFIG.copy()
STABLE_CONFIG_WITH_CO2['learner'] = extend_classifier_for_evaluation(*stable_learner, with_emissions=True)

STABLE_STRING_IDX = UNSTABLE_STRING_IDX = 6

unstable_learner = [grace_period_per_layer(250), add_weight_correction_parameter_to_user_choices, winning_layer_training, vectorized_for_loop, input_preprocessing]
UNSTABLE_CONFIG = {
    'lr': 0.21367378865929126,
    'learner': extend_classifier_for_evaluation(*unstable_learner),
    'layer_weight_learning_rate': 0.002833479724872662,
    'adwin-delta': 0.00044094987538464796,
    'mci': 5.696278201740134e-07,
    'grace_period': 250,
    'grace_type': 'layer_grace',
    'loss_fn': 'NLLLoss',
}

UNSTABLE_CONFIG_WITH_CO2 = UNSTABLE_CONFIG.copy()
UNSTABLE_CONFIG_WITH_CO2['learner'] = extend_classifier_for_evaluation(*unstable_learner, with_emissions=True)

SINGLE_CLASSIFIER_FEATURES_TO_TEST = ['input_preprocessing', 'vectorized', 'winning_layer']
PAIRWISE_CLASSIFIER_FEATURES_TO_TEST = [('delete_deleted_layer', 'disable_deleted_layer')]
classifier_features_to_always_include = ['decoupled_lrs']

singular_combinations = [t for i in range(len(SINGLE_CLASSIFIER_FEATURES_TO_TEST) + 1) for t in combinations(SINGLE_CLASSIFIER_FEATURES_TO_TEST, r=i)]
combs = [ext + s_combi for s_combi in singular_combinations for pair in PAIRWISE_CLASSIFIER_FEATURES_TO_TEST for ext in [x for i in range(2) for x in combinations(pair, r=i)]]
combs.sort(key=lambda x: len(x), reverse=True)
CLASSIFIERS = [combi + (ext,) for combi in combs for ext in classifier_features_to_always_include]

AMOUNT_OF_CLASSIFIERS = 24
AMOUNT_OF_STRINGS = 9

HYPERPARAMETER_KEYS = ['lr', 'layer_weight_learning_rate', 'adwin-delta', 'mci', 'grace']
HYPERPARAMETERS = {
    'lr': [0.999, 0.1, 0.001],
    'layer_weight_learning_rate': [0.999, 0.5, 0.001],
    'adwin-delta': [0.999, 1e-5, 1e-10],
    'mci': [0.1, 1e-8, 1e-12],
    'grace': [(grace_type, count) for grace_type in ['layer_grace', 'global_grace'] for count in [1, 400, 5000]],
}

HYPERPARAMETERS_NAMES = {
    'lr': 'Learning Rate',
    'layer_weight_learning_rate': 'Voting Weight Learning Rate',
    'adwin-delta': 'Adwin-$\\delta$',
    'mci': 'MCI-Cutoff',
    'grace': 'Grace Period',
    'layer_grace': 'Per Layer',
    'global_grace': 'Global'
}

RENAME_VALUE = lambda x: (x[1], HYPERPARAMETERS_NAMES[x[0]]) if isinstance(x, tuple) else x

AMOUNT_HYPERPARAMETERS = list(map(len, HYPERPARAMETERS.values()))
AMOUNT_HYPERPARAMETERS_BEFORE = [sum(AMOUNT_HYPERPARAMETERS[:i]) for i in range(len(AMOUNT_HYPERPARAMETERS))]
TOTAL_AMOUNT_HYPERPARAMETERS = sum(AMOUNT_HYPERPARAMETERS) 
AMOUNT_HYPERPARAMETER_TESTS = AMOUNT_OF_STRINGS * TOTAL_AMOUNT_HYPERPARAMETERS