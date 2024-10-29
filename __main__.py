import random

import numpy as np
import torch
from capymoa.drift.detectors import ADWIN
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream.generator import SEA
from torch import nn

from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner
from tests.resources import random_initialize_model, optimizer_choices

from capymoa.datasets import Electricity
from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator
from capymoa.classifier import AdaptiveRandomForestClassifier, HoeffdingTree, OnlineBagging

if __name__ == "__main__":
    # drift detection testing:
    # -------------------------
    data_stream = np.random.randint(2, size=2000)
    for i in range(999, 2000):
        data_stream[i] = np.random.randint(6, high=12)

    elec_stream = Electricity()
    stream_sea2drift = DriftStream(stream=[SEA(function=1),
                                           AbruptDrift(position=5000),
                                           SEA(function=3),
                                           GradualDrift(position=10000, width=2000),
                                           # GradualDrift(start=9000, end=12000),
                                           SEA(function=1)])

    active_stream = stream_sea2drift

    ob_learner = AdaptiveRandomForestClassifier(schema=active_stream.get_schema(), ensemble_size=10)
    ob_evaluator = ClassificationEvaluator(schema=active_stream.get_schema())
    detector = ADWIN(delta=0.001)

    feature_count = active_stream.schema.get_num_attributes()
    class_count = active_stream.schema.get_num_classes()
    iteration_count = random.randint(10, 250)

    model: AutoDeepLearner = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)
    local_optimizer = create_adl_optimizer(model, optimizer_choices[0], 0.01)

    model = random_initialize_model(model, iteration_count)
    criterion = nn.CrossEntropyLoss()

    i = 0
    while active_stream.has_more_instances() and i < 25_000:
        i += 1
        instance = active_stream.next_instance()
        # prediction = ob_learner.predict(instance)
        # ob_learner.train(instance)

        prediction = model(torch.tensor(instance.x, dtype=torch.float))
        ob_evaluator.update(instance.y_index, torch.argmax(prediction).item())

        loss = criterion(prediction, torch.tensor(instance.y_index))
        loss.backward()

        local_optimizer.step(instance.y_index)
        local_optimizer.zero_grad()

        detector.add_element(ob_evaluator.accuracy())
        if i % 100 == 0:
            print(f'step {i}: loss: {loss}, accuracy: {ob_evaluator.accuracy()}')
        if detector.detected_warning():
            print('Warning for change in data: ' + str(active_stream.get_schema().dataset_name) + ' - at index: ' + str(i))
        if detector.detected_change():
            print('Change detected in data: ' + str(active_stream.get_schema().dataset_name) + ' - at index: ' + str(i))

    print(detector.warning_index)
    print(detector.detection_index)
    
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
    # We could report the cumulative accuracy every window_size instances with the following code, but that is normally not very insightful.
    # display(classificationEvaluatorARF.metrics_per_window())

    # simple pytorch usecase with randomized, predefined model
    # --------------------------------------------------------

    # feature_count = random.randint(1, 10_000)
    # class_count = random.randint(1, 10_000)
    # iteration_count = random.randint(1000, 10_000)
    # 
    # model: AutoDeepLearner = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)
    # 
    # model = random_initialize_model(model, iteration_count)
    # 
    # for optimizer_choice in optimizer_choices:
    #     local_optimizer = create_adl_optimizer(model, optimizer_choice, 0.01)
    # 
    #     criterion = nn.CrossEntropyLoss()
    # 
    #     input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
    #     target = torch.tensor(random.randint(0, class_count - 1))
    #     prediction = model(input)
    # 
    #     loss = criterion(prediction, target)
    #     loss.backward()
    # 
    #     local_optimizer.step(target)
    #     local_optimizer.zero_grad()
