from itertools import combinations

from ADLClassifier import extend_classifier_for_evaluation, grace_period_per_layer, input_preprocessing, vectorized_for_loop, winning_layer_training, add_weight_correction_parameter_to_user_choices

NR_OF_TRIALS = 500
#  todo: reset max instancens
MAX_INSTANCES = 500
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
    'loss_fn': 'NLLLoss'
}

STANDARD_CONFIG_WITH_CO2 = STANDARD_CONFIG
STANDARD_CONFIG_WITH_CO2['learner'] = extend_classifier_for_evaluation(*STANDARD_LEARNER, with_emissions=True)

STABLE_CONFIG = STANDARD_CONFIG_WITH_CO2
STABLE_STRING_IDX = 0
# todo: find unstable config
UNSTABLE_CONFIG = STANDARD_CONFIG_WITH_CO2
UNSTABLE_STRING_IDX = STABLE_STRING_IDX

singular_classifier_features_to_test = ['input_preprocessing', 'vectorized', 'winning_layer']
pairwise_classifier_features_to_test = [('delete_deleted_layer', 'disable_deleted_layer')]
classifier_features_to_always_include = ['decoupled_lrs']

singular_combinations = [t for i in range(len(singular_classifier_features_to_test) + 1) for t in combinations(singular_classifier_features_to_test, r=i)]
combs = [ext + s_combi for s_combi in singular_combinations for pair in pairwise_classifier_features_to_test for ext in [x for i in range(2) for x in combinations(pair, r=i)]]
combs.sort(key=lambda x: len(x), reverse=True)
CLASSIFIERS = [combi + (ext,) for combi in combs for ext in classifier_features_to_always_include]

AMOUNT_OF_CLASSIFIERS = 23
AMOUNT_OF_STRINGS = 9

HYPERPARAMETER_KEYS = ['lr', 'layer_weight_learning_rate', 'adwin-delta', 'mci', 'grace']
HYPERPARAMETERS = {
    'lr': [1, 0.1, 0.001],
    'layer_weight_learning_rate': [1, 0.5, 0.001],
    'adwin-delta': [1, 1e-5, 1e-10],
    'mci': [0.1, 1e-8, 1e-12],
    'grace': [(grace_type, count) for grace_type in ['layer_grace', 'global_grace'] for count in [1, 400, 5000]],
}

AMOUNT_HYPERPARAMETERS = list(map(len, HYPERPARAMETERS.values()))
AMOUNT_HYPERPARAMETERS_BEFORE = [sum(AMOUNT_HYPERPARAMETERS[:i]) for i in range(len(AMOUNT_HYPERPARAMETERS))]
TOTAL_AMOUNT_HYPERPARAMETERS = sum(AMOUNT_HYPERPARAMETERS) 
AMOUNT_HYPERPARAMETER_TESTS = AMOUNT_OF_STRINGS * TOTAL_AMOUNT_HYPERPARAMETERS