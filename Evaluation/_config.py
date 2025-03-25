from itertools import combinations

from ADLClassifier import extend_classifier_for_evaluation

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

STANDARD_CONFIG = {
    'lr': 0.104083,
    'learner': extend_classifier_for_evaluation(),
    'layer_weight_learning_rate': 0.450039,
    'adwin-delta': 3.63053e-05,
    'mci': 9.34633e-07,
    'grace_period': 369,
    'grace_type': 'layer_grace',
    'loss_fn': 'NLLLoss'
}
singular_classifier_features_to_test = ['input_preprocessing', 'vectorized', 'winning_layer']
pairwise_classifier_features_to_test = [('delete_deleted_layer', 'disable_deleted_layer')]
classifier_features_to_always_include = ['decoupled_lrs']

singular_combinations = [t for i in range(len(singular_classifier_features_to_test) + 1) for t in combinations(singular_classifier_features_to_test, r=i)]
combs = [ext + s_combi for s_combi in singular_combinations for pair in pairwise_classifier_features_to_test for ext in [x for i in range(2) for x in combinations(pair, r=i)]]
combs.sort(key=lambda x: len(x), reverse=True)
CLASSIFIERS = [combi + (ext,) for combi in combs for ext in classifier_features_to_always_include]

AMOUNT_OF_CLASSIFIERS = len(CLASSIFIERS)