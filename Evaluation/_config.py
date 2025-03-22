from ADLClassifier import extend_classifier_for_evaluation

NR_OF_TRIALS = 500
MAX_INSTANCES = 100000
MAX_INSTANCES_TEST = MAX_INSTANCES * 2
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

CLASSIFIERS = [
    ('input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs'),
    ('delete_deleted_layer', 'input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs'),
    ('disable_deleted_layer', 'input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs'),
    ('input_preprocessing', 'vectorized', 'decoupled_lrs'),
    ('vectorized', 'winning_layer', 'decoupled_lrs'),
    ('input_preprocessing', 'winning_layer', 'decoupled_lrs'),
    ('input_preprocessing', 'vectorized', 'winning_layer'),
    ('vectorized', 'decoupled_lrs'),
    ('winning_layer', 'decoupled_lrs'),
    ('vectorized', 'winning_layer'),
    ('input_preprocessing', 'decoupled_lrs'),
    ('input_preprocessing', 'vectorized'),
    ('input_preprocessing', 'winning_layer'),
    ('input_preprocessing',),
    ('vectorized',),
    ('winning_layer',),
    ('decoupled_lrs',),
]
