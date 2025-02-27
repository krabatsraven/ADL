# NR_OF_TRIALS = 500
NR_OF_TRIALS = 1
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
