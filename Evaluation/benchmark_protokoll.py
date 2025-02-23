from Evaluation import hyperparameter_search_for_ADL, evaluate_adl_run, evaluate_simple_run, hyperparameter_search_for_SimpleDNN
from Evaluation._config import NR_OF_TRIALS, STREAM_STRINGS

# find best hyperparameters for all streams with adl
runs = []
for stream in STREAM_STRINGS:
    runs.append(hyperparameter_search_for_ADL(nr_of_trials=NR_OF_TRIALS, stream_name=stream))

# evaluate hyperparameters
for run in runs:
    evaluate_adl_run(run)

# compare to simple dnn: set a size that averages the adl node count
runs.clear()
# compare to small simple dnn
for stream in STREAM_STRINGS:
    runs.append(hyperparameter_search_for_SimpleDNN(nr_of_trials=NR_OF_TRIALS, stream_name=stream))

for run in runs:
    evaluate_simple_run(run)

# run best hyperparameter set also with co2 emission
# run best hyperparameter set for different classifier