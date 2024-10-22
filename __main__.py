import random

import numpy as np
import torch
from torch import nn

from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner
from tests.resources import random_initialize_model, optimizer_choices

if __name__ == "__main__":
    feature_count = random.randint(1, 10_000)
    class_count = random.randint(1, 10_000)
    iteration_count = random.randint(1000, 10_000)

    model: AutoDeepLearner = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)

    model = random_initialize_model(model, iteration_count)

    for optimizer_choice in optimizer_choices:
        local_optimizer = create_adl_optimizer(model, optimizer_choice, 0.01)

        criterion = nn.CrossEntropyLoss()

        input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, class_count - 1))
        prediction = model(input)

        loss = criterion(prediction, target)
        loss.backward()

        local_optimizer.step(target)
        local_optimizer.zero_grad()
