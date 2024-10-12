import random
from copy import deepcopy
from typing import Tuple

import pytest
import torch
from torch import nn

import ADLOptimizer
from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner
from tests.resources import random_initialize_model, optimizer_choices
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward
from tests.test_AutoDeepLearner.test_normalise_voting_weights import has_normalised_voting_weights


class TestAdjustWeights:
    @pytest.fixture(scope="class")
    def float_precision_tolerance(self):
        return 10 ** -6

    @pytest.fixture(scope='class')
    def feature_count(self) -> int:
        return random.randint(1, 10_000)

    @pytest.fixture(scope='class')
    def class_count(self) -> int:
        return random.randint(1, 10_000)

    @pytest.fixture(scope='class')
    def iteration_count(self) -> int:
        return random.randint(1000, 10_000)

    @pytest.fixture(scope='class', autouse=True)
    def model(self, feature_count, class_count, iteration_count) -> AutoDeepLearner:

        model = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)

        model = random_initialize_model(model, iteration_count)

        yield model

        del model

    @pytest.fixture(scope='function')
    def optimizer(self, model: AutoDeepLearner, optimizer_choice: type(torch.optim.Optimizer), learning_rate: float) -> torch.optim.Optimizer:
        return create_adl_optimizer(model, optimizer_choice, learning_rate)

    @staticmethod
    def setup_test(model: ADLOptimizer, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int, learning_rate: float) -> ADLOptimizer:
        local_optimizer = create_adl_optimizer(model, optimizer_choice, learning_rate)

        input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, class_count - 1))

        criterion = nn.CrossEntropyLoss()
        prediction = model(input)

        loss = criterion(prediction, target)
        loss.backward()

        local_optimizer.step(target)

        return local_optimizer

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_votes_all_change(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):

        initial_votes = deepcopy(model.voting_weights)

        local_optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count, 0.01)

        weights_are_same = [initial_votes[key] == model.voting_weights[key] for key in model.voting_weights]
        assert not any(weights_are_same), "all the weights should have changed as the layer was either correct or wrong"

        local_optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_voting_weight_correction_factor_all_change(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):
        initial_weight_correction_factors = {int(key): value for key, value in model.weight_correction_factor.items()}

        local_optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count, 0.01)

        initial_weight_correction_factors_are_same = [
            initial_weight_correction_factors[int(key)] == model.get_weight_correction_factor(int(key))
            for key in model.get_voting_weight_keys()
        ]
        assert not any(initial_weight_correction_factors_are_same), \
            "all the weight correction factors should have changed as the layer was either correct or wrong"

        local_optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_votes_still_normalised(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):

        assert has_normalised_voting_weights(model), "the initial model should have a normalised vector"

        local_optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count, 0.01)

        assert has_normalised_voting_weights(model), "the model should have a normalised vector after optimizer step"
        local_optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_adjusting_votes_keeps_forward_functionality_single(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):
        forward_tests = TestAutoDeepLearnerForward()
        local_optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count, 0.01)
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            class_count,
            msg="After optimizer.step(): "
        )
        local_optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_adjusting_votes_keeps_forward_functionality_multiple(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):
        forward_tests = TestAutoDeepLearnerForward()
        batch_size = 1000
        local_optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count, 0.01)
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            class_count,
            batch_size=batch_size,
            msg=f"After optimizer.step() on "
                f"batch size {batch_size}: "
        )
        local_optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    @pytest.mark.parametrize('learning_rate', [0.0001, 0.01, 0.1, 1])
    def test_learning_rate_influences_weight_correction_change_model_one_layer(self,
                                                               feature_count: int,
                                                               class_count: int,
                                                               optimizer_choice: type(torch.optim.Optimizer),
                                                               learning_rate: float
                                                               ):

        model = AutoDeepLearner(feature_count, class_count)
        local_optimizer = create_adl_optimizer(model, optimizer_choice, learning_rate)

        input = torch.rand(model.input_size, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, model.output_size - 1))

        criterion = nn.CrossEntropyLoss()
        prediction = model(input)

        initial_weight_correction_factor = deepcopy(model.get_weight_correction_factor(0))

        loss = criterion(prediction, target)
        loss.backward()

        local_optimizer.step(target)
        local_optimizer.zero_grad()

        if torch.argmax(prediction) == target:
            assert model.get_weight_correction_factor(0) == initial_weight_correction_factor + learning_rate, \
                "the prediction was correct and the weight correction factor should have increased"
        else:
            assert model.get_weight_correction_factor(0) ==  initial_weight_correction_factor - learning_rate,\
                "the prediction was wrong and the weight correction factor should have decreased"

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    @pytest.mark.parametrize('learning_rate', [0.0001, 0.01, 0.1, 1])
    def test_learning_rate_influences_weight_change_model_one_layer(self,
                                                                               feature_count: int,
                                                                               class_count: int,
                                                                               optimizer_choice: type(torch.optim.Optimizer),
                                                                               learning_rate: float
                                                                               ):
        model = AutoDeepLearner(feature_count, class_count)
        local_optimizer = create_adl_optimizer(model, optimizer_choice, learning_rate)

        input = torch.rand(model.input_size, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, model.output_size - 1))

        criterion = nn.CrossEntropyLoss()
        prediction = model(input)

        initial_weight_correction_factor = deepcopy(model.get_weight_correction_factor(0))
        initial_voting_voting_weight = deepcopy(model.get_voting_weight(0))

        loss = criterion(prediction, target)
        loss.backward()

        local_optimizer.step(target)
        local_optimizer.zero_grad()

        if torch.argmax(prediction) == target:
            assert model.get_voting_weight(0) == min(
                (1 + initial_voting_voting_weight * (initial_weight_correction_factor + learning_rate)),
                1
            ), \
                "the prediction was correct and the weight should have increased"
        else:
            assert model.get_voting_weight(0) == initial_voting_voting_weight * (initial_weight_correction_factor - learning_rate), \
                "the prediction was wrong and the weight should have decreased"
