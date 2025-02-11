from Evaluation.RayTuneResources import hyperparameter_search_for_ADL
from Evaluation.RayTuneResources.SimpleDNN import hyperparameter_search_for_SimpleDNN


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

if __name__ == "__main__":
    # _test_example()
    hyperparameter_search_for_ADL(1000, stream_name='electricity')
    hyperparameter_search_for_ADL(1000, stream_name='simple_agraval_drift_back_and_forth')
    hyperparameter_search_for_SimpleDNN(1000, stream_name='electricity')
    hyperparameter_search_for_SimpleDNN(1000, stream_name='simple_agraval_drift_back_and_forth')
