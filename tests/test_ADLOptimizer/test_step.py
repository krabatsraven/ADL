import random
from copy import deepcopy
from itertools import combinations
from typing import Tuple

import pytest
import torch.optim
from torch import nn

from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner

optimizer_choices = [torch.optim.SGD, torch.optim.Adam]
learning_rate_combinations = list(combinations([0.0001, 0.01, 0.1, 1], 2))
trainings_steps = range(2, 5)

@pytest.mark.parametrize('optimizer_choice', optimizer_choices)
class TestOptimizerStep:
    @pytest.fixture(scope='class')
    def feature_count(self) -> int:
        return random.randint(1, 10_000)

    @pytest.fixture(scope='class')
    def class_count(self) -> int:
        return random.randint(1, 10_000)

    @pytest.fixture(scope='class')
    def iteration_count(self) -> int:
        return random.randint(10, 100)

    @pytest.fixture(scope='class')
    def input(self, feature_count: int) -> torch.Tensor:
        return torch.rand(feature_count, requires_grad=True, dtype=torch.float)

    @pytest.fixture(scope='class')
    def target(self, class_count: int) -> torch.Tensor:
        return torch.tensor(random.randint(0, class_count - 1))

    @pytest.fixture
    def criterion(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    @pytest.fixture(autouse=True)
    def model(self,
              feature_count: int,
              class_count: int,
              iteration_count: int
              ) -> AutoDeepLearner:

        model = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)
        model.name = "test step model"

        for _ in range(iteration_count):
            dice = random.choice(range(1, 5))

            match dice:
                case 1:
                    model._add_layer()
                    last_added_layer_idx = len(model.layers) - 1
                    model.voting_weights[last_added_layer_idx] = 1
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

    @pytest.fixture(scope="function")
    def two_same_models(self,
                        model: AutoDeepLearner,
                        feature_count: int,
                        class_count: int
                        ):

        model_1 = AutoDeepLearner(feature_count, class_count)
        model_2 = AutoDeepLearner(feature_count, class_count)

        # both models start at the same point but are independent
        model_1.layers = deepcopy(model.layers)
        model_1.voting_linear_layers = deepcopy(model.voting_linear_layers)
        model_1.voting_weights = deepcopy(model.voting_weights)
        model_2.layers = deepcopy(model.layers)
        model_2.voting_linear_layers = deepcopy(model.voting_linear_layers)
        model_2.voting_weights = deepcopy(model.voting_weights)

        return model_1, model_2

    @pytest.fixture(scope="class")
    def learning_rate(self) -> float:
        return 0.01

    @pytest.fixture(scope="class")
    def nr_of_trainings_steps(self) -> int:
        return 1

    @pytest.fixture(scope='function')
    def optimizer(
            self,
            model: AutoDeepLearner,
            optimizer_choice: type(torch.optim.Optimizer),
            criterion: nn.CrossEntropyLoss,
            input: torch.Tensor,
            target: torch.Tensor,
            learning_rate: float
    ) -> torch.optim.Optimizer:

        opti = create_adl_optimizer(model, optimizer_choice, learning_rate)
        opti.zero_grad()

        yield opti

        opti.zero_grad()

    @pytest.fixture(scope='function')
    def two_optimizer(self,
                      two_same_models: Tuple[AutoDeepLearner, AutoDeepLearner],
                      criterion: nn.CrossEntropyLoss,
                      optimizer_choice: type(torch.optim.Optimizer),
                      small_learning_rate: float,
                      big_learning_rate: float
                      ):

        model_1, model_2 = two_same_models

        small_step_optimizer = create_adl_optimizer(model_1, optimizer_choice, small_learning_rate)
        big_step_optimizer = create_adl_optimizer(model_2, optimizer_choice, big_learning_rate)

        return small_step_optimizer, big_step_optimizer

    @staticmethod
    def optimizer_step(local_model: AutoDeepLearner,
                       local_criterion: nn.CrossEntropyLoss,
                       local_optimizer: torch.optim.Optimizer,
                       local_input: torch.Tensor,
                       local_target: torch.Tensor
                       ) -> None:

        local_prediction = local_model(local_input)
        local_loss = local_criterion(local_prediction, local_target)
        local_loss.backward()
        local_optimizer.step(local_target)
        local_optimizer.zero_grad()

    def test_is_correct_optimizer(
            self,
            model: AutoDeepLearner,
            optimizer_choice: type(torch.optim.Optimizer)
    ):
        build_optimizer = create_adl_optimizer(model, optimizer_choice, 0.01)

        assert build_optimizer.__repr__().startswith("ADLOptimizer"), "the optimizer should be an ADLOptimizer"
        assert isinstance(build_optimizer, optimizer_choice), \
            "The created object should stil be an instance of the chosen optimizer class"

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    def test_step_changes_hidden_layer_weights(
            self,
            model: AutoDeepLearner,
            optimizer: torch.optim.Optimizer,
            input: torch.Tensor,
            target: torch.Tensor,
            criterion: nn.CrossEntropyLoss,
            nr_of_trainings_steps: int,
    ):

        initial_models_hidden_layer_weights = [
            hidden_layer.weight.clone()
            for hidden_layer in model.layers
        ]

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model, criterion, optimizer, input, target)

        hidden_layers_weights_are_not_changing = all(
            torch.equal(
                initial_hidden_layer_weight,
                current_hidden_layer.weight
            )
            for initial_hidden_layer_weight, current_hidden_layer
            in zip(initial_models_hidden_layer_weights, model.layers))

        assert not hidden_layers_weights_are_not_changing,\
            "the hidden layers should be changing their weights if optimizing is happening"

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    def test_step_changes_hidden_layer_biases(
            self,
            model: AutoDeepLearner,
            optimizer: torch.optim.Optimizer,
            input: torch.Tensor,
            target: torch.Tensor,
            criterion: nn.CrossEntropyLoss,
            nr_of_trainings_steps: int,
    ):

        initial_models_hidden_layer_biases = [
            hidden_layer.bias.clone()
            for hidden_layer in model.layers
        ]

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model, criterion, optimizer, input, target)

        hidden_layers_biases_are_not_changing = all(
            torch.equal(
                initial_models_hidden_layer_bias,
                current_hidden_layer.bias
            )
            for initial_models_hidden_layer_bias, current_hidden_layer
            in zip(initial_models_hidden_layer_biases, model.layers))

        assert not hidden_layers_biases_are_not_changing, \
            "the hidden layers should be changing their biases if optimizing is happening"

        optimizer.zero_grad()

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    def test_step_changes_output_layer_weights(
            self,
            model: AutoDeepLearner,
            optimizer: torch.optim.Optimizer,
            input: torch.Tensor,
            target: torch.Tensor,
            criterion: nn.CrossEntropyLoss,
            nr_of_trainings_steps: int,
    ):

        initial_models_output_layer_weights = [
            output_layer.weight.clone()
            for output_layer in model.voting_linear_layers.values()
        ]

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model, criterion, optimizer, input, target)

        output_layers_weights_are_not_changing = all(
            torch.equal(
                initial_output_layer_weight,
                current_output_layer.weight
            )
            for initial_output_layer_weight, current_output_layer
            in zip(initial_models_output_layer_weights, model.voting_linear_layers.values()))

        assert not output_layers_weights_are_not_changing, \
            "the output layers should be changing their weights if optimizing is happening"

        optimizer.zero_grad()

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    def test_step_changes_output_layer_biases(
            self,
            model: AutoDeepLearner,
            optimizer: torch.optim.Optimizer,
            input: torch.Tensor,
            target: torch.Tensor,
            criterion: nn.CrossEntropyLoss,
            nr_of_trainings_steps: int,
    ):

        initial_models_output_layer_biases = [
            output_layer.bias.clone()
            for output_layer in model.voting_linear_layers.values()
        ]

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model, criterion, optimizer, input, target)

        output_layers_biases_are_not_changing = all(
            torch.equal(
                initial_output_layer_bias,
                current_output_layer.bias
            )
            for initial_output_layer_bias, current_output_layer
            in zip(initial_models_output_layer_biases, model.voting_linear_layers.values()))

        assert not output_layers_biases_are_not_changing,\
            "the output layers should be changing their biases if optimizing is happening"

        optimizer.zero_grad()

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    @pytest.mark.parametrize('small_learning_rate,big_learning_rate', learning_rate_combinations)
    def test_learning_rate_effects_changes_of_hidden_layers_weights(
            self,
            two_same_models: Tuple[AutoDeepLearner, AutoDeepLearner],
            criterion: nn.CrossEntropyLoss,
            two_optimizer: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
            input: torch.Tensor,
            target: torch.Tensor,
            nr_of_trainings_steps: int,
    ):

        model_1, model_2 = two_same_models
        small_step_optimizer, big_step_optimizer = two_optimizer

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model_1, criterion, small_step_optimizer, input, target)
            self.optimizer_step(model_2, criterion, big_step_optimizer, input, target)

        hidden_layers_are_the_same_with_both_models = all(
            torch.equal(
                model_1_hidden_layer.weight,
                model_2_hidden_layer.weight
            )
            for model_1_hidden_layer, model_2_hidden_layer
            in zip(model_1.layers, model_2.layers))

        assert not hidden_layers_are_the_same_with_both_models, \
            "the weights of the hidden layers should be changing differently for different learning rates"

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    @pytest.mark.parametrize('small_learning_rate,big_learning_rate', learning_rate_combinations)
    def test_learning_rate_effects_changes_of_output_layers_weights(
            self,
            two_same_models: Tuple[AutoDeepLearner, AutoDeepLearner],
            criterion: nn.CrossEntropyLoss,
            two_optimizer: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
            input: torch.Tensor,
            target: torch.Tensor,
            nr_of_trainings_steps: int,
    ):

        model_1, model_2 = two_same_models
        small_step_optimizer, big_step_optimizer = two_optimizer

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model_1, criterion, small_step_optimizer, input, target)
            self.optimizer_step(model_2, criterion, big_step_optimizer, input, target)

        output_layers_are_the_same_with_both_models = all(
            torch.equal(
                model_1_hidden_layer.weight,
                model_2_hidden_layer.weight
            )
            for model_1_hidden_layer, model_2_hidden_layer
            in zip(model_1.voting_linear_layers.values(), model_2.voting_linear_layers.values()))

        assert not output_layers_are_the_same_with_both_models, \
            "the weights of the output layers should be changing differently for different learning rates"

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    @pytest.mark.parametrize('small_learning_rate,big_learning_rate', learning_rate_combinations)
    def test_learning_rate_effects_changes_of_hidden_layers_biases(
            self,
            two_same_models: Tuple[AutoDeepLearner, AutoDeepLearner],
            criterion: nn.CrossEntropyLoss,
            two_optimizer: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
            input: torch.Tensor,
            target: torch.Tensor,
            nr_of_trainings_steps: int,
    ):

        model_1, model_2 = two_same_models
        small_step_optimizer, big_step_optimizer = two_optimizer

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model_1, criterion, small_step_optimizer, input, target)
            self.optimizer_step(model_2, criterion, big_step_optimizer, input, target)

        hidden_layers_are_the_same_with_both_models = all(
            torch.equal(
                model_1_hidden_layer.bias,
                model_2_hidden_layer.bias
            )
            for model_1_hidden_layer, model_2_hidden_layer
            in zip(model_1.layers, model_2.layers))

        assert not hidden_layers_are_the_same_with_both_models, \
            "the biases of the hidden layers should be changing differently for different learning rates"

    @pytest.mark.parametrize('nr_of_trainings_steps', trainings_steps)
    @pytest.mark.parametrize('small_learning_rate,big_learning_rate', learning_rate_combinations)
    def test_learning_rate_effects_changes_of_output_layers_biases(
            self,
            two_same_models: Tuple[AutoDeepLearner, AutoDeepLearner],
            criterion: nn.CrossEntropyLoss,
            two_optimizer: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
            input: torch.Tensor,
            target: torch.Tensor,
            nr_of_trainings_steps: int,
    ):

        model_1, model_2 = two_same_models
        small_step_optimizer, big_step_optimizer = two_optimizer

        for _ in range(nr_of_trainings_steps):
            self.optimizer_step(model_1, criterion, small_step_optimizer, input, target)
            self.optimizer_step(model_2, criterion, big_step_optimizer, input, target)

        output_layers_are_the_same_with_both_models = all(
            torch.equal(
                model_1_hidden_layer.bias,
                model_2_hidden_layer.bias
            )
            for model_1_hidden_layer, model_2_hidden_layer
            in zip(model_1.voting_linear_layers.values(), model_2.voting_linear_layers.values()))

        assert not output_layers_are_the_same_with_both_models, \
            "the biases of the output layers should be changing differently for different learning rates"
