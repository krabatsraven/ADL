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
    runs = []
    runs.append(hyperparameter_search_for_ADL(1000, stream_name='electricity'))
    runs.append(hyperparameter_search_for_ADL(1000, stream_name='simple_agraval_single_drift'))
    runs.append(hyperparameter_search_for_ADL(1000, stream_name='simple_agraval_three_drifts'))
    runs.append(hyperparameter_search_for_ADL(1000, stream_name='simple_agraval_drift_back_and_forth'))
    for run in runs:
        evaluate_adl_run(run)
    runs.clear()
    runs.append(hyperparameter_search_for_SimpleDNN(1000, stream_name='electricity'))
    runs.append(hyperparameter_search_for_SimpleDNN(1000, stream_name='simple_agraval_single_drift'))
    runs.append(hyperparameter_search_for_SimpleDNN(1000, stream_name='simple_agraval_three_drifts'))
    runs.append(hyperparameter_search_for_SimpleDNN(1000, stream_name='simple_agraval_drift_back_and_forth'))
    for run in runs:
        evaluate_adl_run(run)
