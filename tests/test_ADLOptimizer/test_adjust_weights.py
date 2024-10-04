import random
from copy import deepcopy

import pytest
import torch
from torch import nn

import ADLOptimizer
from ADLOptimizer import create_adl_optimizer
from AutoDeepLearner import AutoDeepLearner
from tests.test_ADLOptimizer.test_step import optimizer_choices
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward
from tests.test_AutoDeepLearner.test_normalise_voting_weights import has_normalised_voting_weights


class TestAdjustWeights:
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
    def model(self, feature_count, class_count, iteration_count) -> AutoDeepLearner:

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

    @staticmethod
    def setup_test(model: ADLOptimizer, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int) -> ADLOptimizer:
        optimizer = create_adl_optimizer(model, optimizer_choice)

        input = torch.rand(feature_count, requires_grad=True, dtype=torch.float)
        target = torch.tensor(random.randint(0, class_count - 1))


        criterion = nn.CrossEntropyLoss()
        prediction = model(input)

        loss = criterion(prediction, target)
        loss.backward()

        optimizer.step(target)

        return optimizer

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_votes_all_change(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):

        initial_votes = deepcopy(model.voting_weights)

        optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count)

        weights_are_same = [initial_votes[key] == model.voting_weights[key] for key in model.voting_weights]
        assert not any(weights_are_same), "all the weights should have changed as the layer was either correct or wrong"

        optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_voting_weight_correction_factor_all_change(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):
        initial_weight_correction_factors = deepcopy(model.weight_correction_factor)

        optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count)

        initial_weight_correction_factors_are_same = [
            initial_weight_correction_factors[key] == model.weight_correction_factor[key]
            for key in model.voting_weights
        ]
        assert not any(initial_weight_correction_factors_are_same), \
            "all the weight correction factors should have changed as the layer was either correct or wrong"

        optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_votes_still_normalised(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):

        assert has_normalised_voting_weights(model), "the initial model should have a normalised vector"

        optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count)

        assert has_normalised_voting_weights(model), "the model should have a normalised vector after optimizer step"
        optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_adjusting_votes_keeps_forward_functionality_single(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):
        forward_tests = TestAutoDeepLearnerForward()
        optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count)
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            class_count,
            msg="After optimizer.step(): "
        )
        optimizer.zero_grad()

    @pytest.mark.parametrize('optimizer_choice', optimizer_choices)
    def test_adjusting_votes_keeps_forward_functionality_multiple(self, model: AutoDeepLearner, optimizer_choice: torch.optim.Optimizer, feature_count: int, class_count: int):
        forward_tests = TestAutoDeepLearnerForward()
        batch_size = 1000
        optimizer = self.setup_test(model, optimizer_choice, feature_count, class_count)
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            class_count,
            batch_size=batch_size,
            msg=f"After optimizer.step() on "
                f"batch size {batch_size}: "
        )
        optimizer.zero_grad()