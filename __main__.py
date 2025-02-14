from Evaluation.EvaluationFunctions import _test_example

from Evaluation import evaluate_adl_run, evaluate_simple_run, hyperparameter_search_for_SimpleDNN, hyperparameter_search_for_ADL

# todo: evaluate: # simple dnn:
#  {
#  'lr': 0.008207173029285171,
#  'model_structure': [256, 256, 2048, 1024],
#  'stream': '/home/david/PycharmProjects/ADL/data/electricity.arff'
#  }

# {'lr': 0.12898917713611163, 'model_structure': [512, 1024, 512], 'stream': 'simple_agraval_drift_back_and_forth'}

# todo: evaluate adl:
#  {
#  'learner': ('vectorized', 'winning_layer'),
#  'stream': 'electricity',
#  'lr': 0.3229505539221693,
#  'adwin-delta': 1.6181109476398546e-06,
#  'mci': 8.993030033597217e-06,
#  'grace_period': (8, 'global_grace'),
#  'loss_fn': <function <lambda> at 0x7f9f2e6af880>}

# 'learner': ('vectorized', 'winning_layer'), 'stream': 'electricity', 'lr': 0.4582336518626576, 'adwin-delta': 3.4331309627768405e-05, 'mci': 3.0284480025946798e-06, 'grace_period': (4, 'layer_grace'), 'loss_fn': CrossEntropyLoss()}

if __name__ == "__main__":
    # _test_example()
    stream_strings = [
        'electricity',
        'agraval_no_drift', 'agraval_single_drift', 'agraval_three_drifts', 'agraval_drift_back_and_forth',
        'sea_no_drift', 'sea_single_drift', 'sea_three_drifts', 'sea_drift_back_and_forth'
    ]
    runs = []
    for stream_name in stream_strings:
        runs.append(hyperparameter_search_for_ADL(500, stream_name=stream_name))
    for run in runs:
        evaluate_adl_run(run)
    runs.clear()

    for stream_name in stream_strings:
        runs.append(hyperparameter_search_for_SimpleDNN(500, stream_name=stream_name))
    for run in runs:
        evaluate_simple_run(run)
