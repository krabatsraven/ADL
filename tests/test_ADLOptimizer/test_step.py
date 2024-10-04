import random

import pytest
import torch.optim
from torch import nn

from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner

optimizer_choices = [torch.optim.SGD, torch.optim.Adam]


class TestOptimizerStep:
    @pytest.fixture(scope="class")
    def float_precision_tolerance(self):
        return 10 ** -6

    @pytest.fixture(scope='class')
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope='class')
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope='class')
    def iteration_count(self) -> int:
        return random.randint(1000, 10_000)

    @pytest.fixture(scope='class', autouse=True)
    def model(self, feature_count: int, class_count: int, iteration_count: int) -> AutoDeepLearner:

        model = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)

        for _ in range(iteration_count):
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

        yield model

        del model

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_is_correct_optimizer(self, model: AutoDeepLearner, optimizer_choice: type(torch.optim.Optimizer)):
        optimizer = create_adl_optimizer(model, optimizer_choice)

        assert optimizer.__repr__().startswith("ADLOptimizer"), "the optimizer should be an ADLOptimizer"
        assert isinstance(optimizer, optimizer_choice), \
            "The created object should stil be an instance of the chosen optimizer class"

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_step_changes_parameter(
            self,
            model: AutoDeepLearner,
            optimizer_choice: type(torch.optim.Optimizer),
            feature_count: int,
            class_count: int
    ):
        optimizer = create_adl_optimizer(model, optimizer_choice)

        input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, class_count - 1))

        initial_params = [param.clone() for param in model.parameters()]

        criterion = nn.CrossEntropyLoss()
        prediction = model(input)

        loss = criterion(prediction, target)
        loss.backward()


        optimizer.step(target)

        for initial_param, param in zip(initial_params, model.parameters()):
            assert not torch.equal(initial_param, param), \
                f"parameters should have changed after step but initial:({initial_param}) == current:({param})"

        optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_learning_rate_effects_changes_of_parameters(
            self,
            model: AutoDeepLearner,
            optimizer_choice: type(torch.optim.Optimizer),
            feature_count: int,
            class_count: int
    ):

        criterion = nn.CrossEntropyLoss()

        input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, class_count - 1))


        learning_results = {
            'initial parameters': [param.clone() for param in model.parameters()]
        }

        for learning_rate in [0.0001, 0.01, 0.1, 1, 10]:

            prediction = model(input)
            optimizer = create_adl_optimizer(network=model, optimizer=optimizer_choice, learning_rate=learning_rate)

            loss = criterion(prediction, target)
            loss.backward()

            optimizer.step(target)

            learning_results[f'learning rate={learning_rate}'] =  [param.clone() for param in model.parameters()]
            optimizer.zero_grad()

        initial_params = learning_results['initial parameters']
        delta_learning_result = dict()

        learning_rates = [0.0001, 0.01, 0.1, 1, 10]
        for learning_rate in learning_rates:
            delta_learning_result[f'learning rate={learning_rate}'] = [
                torch.subtract(param, initial_param)
                for initial_param, param in zip(initial_params, learning_results[f'learning rate={learning_rate}'])
            ]

        deltas_grow_compared_to_previous_learning_rate = [
            torch.all(torch.greater(torch.subtract(
                param_from_bigger_lr_difference,
                param_from_smaller_lr_difference
            ), torch.tensor(0))) 
            for i in range(1, len(learning_rates))
            for param_from_bigger_lr_difference, param_from_smaller_lr_difference in zip(
                delta_learning_result[f'learning rate={learning_rates[i]}'],
                delta_learning_result[f'learning rate={learning_rates[i - 1]}']
            )
        ]
        assert all(deltas_grow_compared_to_previous_learning_rate), \
            (f"with growing learning rates the difference in changes to the initial parameters should grow, "
             f"instead: {[f'learning rate={learning_rate}: {comparison}' for learning_rate, comparison in zip(learning_rates, deltas_grow_compared_to_previous_learning_rate)]}")