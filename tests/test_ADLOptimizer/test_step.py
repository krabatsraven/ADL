import random

import pytest
import torch.optim
from torch import nn

from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_integration_internal_methods import TestAutoDeepLearnerIntegration

optimizer_choices = [torch.optim.SGD, torch.optim.Adam]


class TestOptimizerStep:
    @pytest.fixture(scope='class', autouse=True)
    def model(self) -> AutoDeepLearner:

        model = AutoDeepLearner(nr_of_features=10, nr_of_classes=7)

        for _ in range(1000):
            dice = random.choice(range(1, 5))

            match dice:
                case 1:
                    model._add_layer()
                    last_added_layer_idx = len(model.layers) - 1
                    model.voting_weights[last_added_layer_idx] = random.uniform(0, 1)
                    model._normalise_voting_weights()
                case 2:
                    layer_choice = random.choice(list(model.voting_weights.keys()))
                    model._add_node(layer_choice)
                case 3:
                    if len(model.voting_weights.keys()) > 2:
                        layer_choice = random.choice(list(model.voting_weights.keys()))
                        model._prune_layer_by_vote_removal(layer_choice)
                    else:
                        continue
                case 4:
                    if len(model.voting_weights.keys()) > 2:
                        layer_choice = random.choice(list(model.voting_weights.keys()))
                        if model.layers[layer_choice].weight.size()[0] > 2:
                            node_choice = random.choice(range(model.layers[layer_choice].weight.size()[0]))
                            model._delete_node(layer_choice, node_choice)
                        else:
                            continue
                    else:
                        continue

        return model

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_is_correct_optimizer(self, model, optimizer_choice):
        optimizer = create_adl_optimizer(model, optimizer_choice)

        assert optimizer.__repr__().startswith("ADLOptimizer"), "the optimizer should be an ADLOptimizer"
        assert isinstance(optimizer, optimizer_choice), \
            "The created object should stil be an instance of the chosen optimizer class"

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_step_changes_parameter(self, model, optimizer_choice):
        torch.set_grad_enabled(True)
        optimizer = create_adl_optimizer(model, optimizer_choice)

        input = torch.rand(10, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, 6))

        criterion = nn.CrossEntropyLoss()
        prediction = model(input)

        loss = criterion(prediction, target)

        loss.backward()

        initial_params = [param.clone() for param in model.parameters()]

        optimizer.step(target)

        for initial_param, param in zip(initial_params, model.parameters()):
            assert not torch.equal(initial_param, param), \
                f"parameters should have changed after step but initial:({initial_param}) == current:({param})"

        optimizer.zero_grad()
