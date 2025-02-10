from Evaluation import simple_agraval_single_drift, simple_agraval_three_drifts, simple_agraval_drift_back_and_forth
from Evaluation.EvaluationFunctions import _test_example

from Evaluation.RayTuneResources import hyperparameter_search_for_ADL
from Evaluation.RayTuneResources.SimpleDNN import hyperparameter_search_for_SimpleDNN


# todo: evaluate: # simple dnn: 
#  {
#  'lr': 0.008207173029285171, 
#  'model_structure': [256, 256, 2048, 1024], 
#  'stream': '/home/david/PycharmProjects/ADL/data/electricity.arff'
#  }


if __name__ == "__main__":
    # _test_example()
    hyperparameter_search_for_ADL(10)
    hyperparameter_search_for_SimpleDNN(10)
    simple_agraval_single_drift.save('/home/david/PycharmProjects/ADL/data/simple_agraval_single_drift')
    simple_agraval_three_drifts.save('/home/david/PycharmProjects/ADL/data/simple_agraval_three_drifts')
    simple_agraval_drift_back_and_forth.save('/home/david/PycharmProjects/ADL/data/simple_agraval_drift_back_and_forth')
