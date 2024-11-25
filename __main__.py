import random
import time

import numpy as np
import torch
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream.generator import SEA
from torch import nn
from torch.optim import SGD

from ADLClassifier import ADLClassifier
from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner
from tests.resources import random_initialize_model, optimizer_choices

from capymoa.datasets import Electricity, ElectricityTiny
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator, prequential_evaluation
from capymoa.classifier import AdaptiveRandomForestClassifier, HoeffdingTree, OnlineBagging

if __name__ == "__main__":

    ### capymoa training of classifier on a stream:
    ### -------------------------------------------
    elec_stream = ElectricityTiny()
    adl_classifier = ADLClassifier(schema=elec_stream.schema)

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=elec_stream, learner=adl_classifier, window_size=100, optimise=True, store_predictions=False, store_y=False)
    total_time_end = time.time_ns()

    print(f"total time spend in covariance loop: {adl_classifier.total_time_in_loop:.2E}ns, that equals {adl_classifier.total_time_in_loop / 10 ** 9:.2f}s or {adl_classifier.total_time_in_loop / 10 ** 9 /60:.2}min")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"meaning that the covariance loop alone took {adl_classifier.total_time_in_loop / (total_time_end - total_time_start) * 100}% of the training time")

    print("\tDifferent ways of accessing metrics:")

    print(f"results_ht['wallclock']: {results_ht['wallclock']} results_ht.wallclock(): {results_ht.wallclock()}")
    print(f"results_ht['cpu_time']: {results_ht['cpu_time']} results_ht.cpu_time(): {results_ht.cpu_time()}")

    print(f"results_ht.cumulative.accuracy() = {results_ht.cumulative.accuracy()}")
    print(f"results_ht.cumulative['accuracy'] = {results_ht.cumulative['accuracy']}")
    print(f"results_ht['cumulative'].accuracy() = {results_ht['cumulative'].accuracy()}")
    print(f"results_ht.accuracy() = {results_ht.accuracy()}")

    print(f"\n\tAll the cumulative results:")
    print(results_ht.cumulative.metrics_dict())

    print(f"\n\tAll the windowed results:")
    # display(results_ht.metrics_per_window())
    # OR display(results_ht.windowed.metrics_per_window())

    # results_ht.write_to_file() -> this will save the results to a directory

    plot_windowed_results(results_ht, metric= "accuracy")

    print(f"the learner has {len(adl_classifier.model.layers)} hidden layers and {len(adl_classifier.model.voting_linear_layers)} output layers")


    # # drift detection testing:
    # # -------------------------
    # data_stream = np.random.randint(2, size=2000)
    # for i in range(999, 2000):
    #     data_stream[i] = np.random.randint(6, high=12)
    # 
    # elec_stream = Electricity()
    # stream_sea2drift = DriftStream(stream=[SEA(function=1),
    #                                        AbruptDrift(position=5000),
    #                                        SEA(function=3),
    #                                        GradualDrift(position=10000, width=2000),
    #                                        # GradualDrift(start=9000, end=12000),
    #                                        SEA(function=1)])
    # 
    # active_stream = stream_sea2drift
    # 
    # ob_learner = AdaptiveRandomForestClassifier(schema=active_stream.get_schema(), ensemble_size=10)
    # ob_evaluator = ClassificationEvaluator(schema=active_stream.get_schema())
    # detector = ADWIN(delta=0.001)
    # 
    # feature_count = active_stream.schema.get_num_attributes()
    # class_count = active_stream.schema.get_num_classes()
    # iteration_count = random.randint(10, 250)
    # 
    # model: AutoDeepLearner = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)
    # local_optimizer = create_adl_optimizer(model, optimizer_choices[0], 0.01)
    # 
    # model = random_initialize_model(model, iteration_count)
    # criterion = nn.CrossEntropyLoss()
    # 
    # i = 0
    # while active_stream.has_more_instances() and i < 25_000:
    #     i += 1
    #     instance = active_stream.next_instance()
    #     # prediction = ob_learner.predict(instance)
    #     # ob_learner.train(instance)
    # 
    #     prediction = model(torch.tensor(instance.x, dtype=torch.float))
    #     ob_evaluator.update(instance.y_index, torch.argmax(prediction).item())
    # 
    #     loss = criterion(prediction, torch.tensor(instance.y_index))
    #     loss.backward()
    # 
    #     local_optimizer.step(instance.y_index)
    #     local_optimizer.zero_grad()
    # 
    #     detector.add_element(loss)
    #     if i % 100 == 0:
    #         print(f'step {i}: loss: {loss}, accuracy: {ob_evaluator.accuracy()}')
    #     if detector.detected_warning():
    #         print('Warning for change in data: ' + str(active_stream.get_schema().dataset_name) + ' - at index: ' + str(i))
    #     if detector.detected_change():
    #         print('Change detected in data: ' + str(active_stream.get_schema().dataset_name) + ' - at index: ' + str(i))
    # 
    # print(detector.warning_index)
    # print(detector.detection_index)

    # capymoa hello world:
    # -------------------------

    # stream = Electricity()
    # 
    # ARF = AdaptiveRandomForestClassifier(schema=stream.get_schema(), ensemble_size=10)
    # 
    # # The window_size in ClassificationWindowedEvaluator specifies the amount of instances used per evaluation
    # windowedEvaluatorARF = ClassificationWindowedEvaluator(schema=stream.get_schema(), window_size=4500)
    # # The window_size ClassificationEvaluator just specifies the frequency at which the cumulative metrics are stored
    # classificationEvaluatorARF = ClassificationEvaluator(schema=stream.get_schema(), window_size=4500)
    # 
    # while stream.has_more_instances():
    #     instance = stream.next_instance()
    #     prediction = ARF.predict(instance)
    #     windowedEvaluatorARF.update(instance.y_index, prediction)
    #     classificationEvaluatorARF.update(instance.y_index, prediction)
    #     ARF.train(instance)
    # 
    # # Showing only the 'classifications correct (percent)' (i.e. accuracy)
    # print(f'[ClassificationWindowedEvaluator] Windowed accuracy reported for every window_size windows')
    # print(windowedEvaluatorARF.accuracy())
    # 
    # print(f'[ClassificationEvaluator] Cumulative accuracy: {classificationEvaluatorARF.accuracy()}')
    # # We could report the cumulative accuracy every window_size instances with the following code, but that is normally not very insightful.
    # display(classificationEvaluatorARF.metrics_per_window())

    # simple pytorch usecase with adding a layer to model and optimizer
    # --------------------------------------------------------

    # feature_count = 5
    # class_count = 2
    # iteration_count = random.randint(1000, 10_000)
    #
    # model: AutoDeepLearner = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)
    # optimizer = SGD(model.parameters(), lr=0.01)
    # copy_of_parameters = [param.clone() for param in list(model.parameters())]
    #
    # for i in range(5):
    #     input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
    #     target = torch.tensor(random.randint(0, class_count - 1))
    #     prediction = model(input)
    #     criterion = nn.CrossEntropyLoss()
    #     loss = criterion(prediction, target)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"----------loop {i}------------------------")
    #     print(f"----------parameters loop {i}-------------")
    #     print(optimizer.param_groups[0]['params'])
    #     print(f"-----------these are the same as the models: {optimizer.param_groups[0]['params'] == list(model.parameters())}--------------------")
    #     print(f"------------nr of layers: {len(model.layers)}--------")
    #     print(f"------------nr of parameters: {len(list(optimizer.param_groups[0]['params']))}--------")
    #     if copy_of_parameters is not None:
    #         print(f"------------the parameters have not changed since the last time: {all([torch.isclose(copy, original).all() for copy, original in zip(copy_of_parameters, list(model.parameters()))])}-----------------")
    #     print("-----------end loop-----------------------")
    #     print()
    #     optimizer.zero_grad()
    #     model._add_layer()
    #     optimizer.param_groups[0]['params'] = list(model.parameters())
    #     copy_of_parameters = [param.clone() for param in list(model.parameters())]

