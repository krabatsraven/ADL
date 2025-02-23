# import pathlib
# import time
# 
# import ray
# from capymoa.classifier import AdaptiveRandomForestClassifier, HoeffdingTree, PassiveAggressiveClassifier
# from capymoa.evaluation import prequential_evaluation
# 
# from Evaluation.EvaluationFunctions import _test_example
# 
# from Evaluation import evaluate_adl_run, evaluate_simple_run, hyperparameter_search_for_SimpleDNN, \
#     hyperparameter_search_for_ADL, MAX_INSTANCES, __compare_results_via_plot_and_save
# from Evaluation.RayTuneResources.SimpleDNN.hyperparameter_search_for_SimpleDNN import compare_simple_to_adl, evaluate_comparision_to_adl, get_simple_arguments
# from Evaluation.RayTuneResources.config_handling import load_config, evaluate_adl_run_config, config_to_stream
# from Evaluation._config import NR_OF_TRIALS, MAX_INSTANCES_TEST
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer

from Evaluation.EvaluationFunctions import _test_example
from Evaluation.RayTuneResources.config_handling import config_to_stream

# 82% acc: {'learner': ('vectorized', 'winning_layer', 'decoupled_lrs'), 'stream': 'electricity', 'lr': 0.17037433308206834, 'layer_weight_learning_rate': 0.0051048969488651065, 'adwin-delta': 2.2019797256079463e-05, 'mci': 2.3105218391180886e-07, 'grace_period': (32, 'global_grace'), 'loss_fn': 'NLLLoss'}

if __name__ == "__main__":
    import torch
    import numpy as np
    from capymoa.stream import ARFFStream
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    stream_strings = [
        'electricity',
        'agraval_no_drift', 'agraval_single_drift', 'agraval_three_drifts', 'agraval_drift_back_and_forth',
        'sea_no_drift', 'sea_single_drift', 'sea_three_drifts', 'sea_drift_back_and_forth'
    ]
    # class StreamingStandardScaler(StandardScaler):
    # 
    #     def fit(self, X, y=None, sample_weight=None):
    #         return self.partial_fit(X, y, sample_weight=sample_weight)
    # 
    #     def transform(self, X, copy=None):
    #         return super().transform(X, copy)
    # 
    # 
    # for stream_name in stream_strings:
    #     stream = config_to_stream(stream_name)
    # 
    #     nominal_indicies = [i for i in range(stream.schema.get_num_attributes()) if stream.schema.get_moa_header().attribute(i).isNominal()]
    #     numerical_indicies = [i for i in range(stream.schema.get_num_attributes()) if stream.schema.get_moa_header().attribute(i).isNumeric()]
    #     nominal_values = [torch.arange(len(stream.schema.get_moa_header().attribute(i).getAttributeValues()), dtype=torch.float32) for i in nominal_indicies]
    #     scaler = StandardScaler()
    #     transformer = make_column_transformer(
    #         (OneHotEncoder(categories=nominal_values), nominal_indicies),
    #         (StreamingStandardScaler(), numerical_indicies),
    #         remainder='passthrough',
    #         sparse_threshold=0)
    #     i = 0
    #     while i < 10:
    #         current_instance = stream.next_instance()
    #         if len(current_instance.x.shape) < 2:
    #             x = current_instance.x[np.newaxis, ...]
    #         else:
    #             x = current_instance.x
    # 
    #         if i == 0:
    #             transformer.fit(x)
    #         transformed_nominal = transformer.transform(x)
    #         out_tensor = torch.from_numpy(transformed_nominal)
    # 
    #         i += 1

    _test_example(name="test_norm_and_one_hot_encoding")

    # runs = []
    # classifiers = [AdaptiveRandomForestClassifier]
    # for stream_name in stream_strings:
    #     for classifier in classifiers:
    #         stream_data = config_to_stream(stream_name)
    #         learner = classifier(schema=stream_data.schema)
    #         print(f"Stream: {stream_name}, Classifier: {classifier.__name__}")
    #         total_time_start = time.time_ns()
    #         results_ht = prequential_evaluation(stream=stream_data, learner=learner, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES_TEST)
    #         total_time_end = time.time_ns()
    # 
    #         print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    # 
    #         print(f"\n\tThe cumulative results:")
    #         print(f"instances={results_ht.cumulative.metrics_dict()['instances']}, accuracy={results_ht.cumulative.metrics_dict()['accuracy']}")
    #         print("----------------------------------------------------\n")
    # 
    # # todo: fix that multiple comparisions overwrite each other
    # # print("comparing to adl runs 8")
    # # runs.clear()
    # # tasks = get_simple_arguments(run_id=8, path_to_summary=(pathlib.Path(f'/home/david/PycharmProjects/ADL/results/runs/runID={run}/summary.csv')), nr_of_trials=NR_OF_TRIALS)
    # # for task in tasks:
    # #     runs.append(hyperparameter_search_for_SimpleDNN(**task))
    # # print("evaluating the simple runs")
    # # for run in runs:
    # #     evaluate_comparision_to_adl(run)