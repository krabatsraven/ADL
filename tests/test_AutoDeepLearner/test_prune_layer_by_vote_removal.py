import math
import random

import pytest
from PIL.features import modules

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


class TestPruneLayerByVoteRemoval:
    @pytest.fixture(scope="class")
    def float_precision_tolerance(self):
        return 10 ** -6

    @pytest.fixture(scope='class')
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope='class')
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope='class', autouse=True)
    def nr_of_layers(self) -> int:
        return 100

    @pytest.fixture(autouse=True)
    def model(self, feature_count, class_count, nr_of_layers) -> AutoDeepLearner:
        model = AutoDeepLearner(feature_count, class_count)
        model.voting_weights[0] = random.randint(0, 100)

        for i in range(nr_of_layers):
            model._add_layer()

            model.voting_weights[i + 1] = random.randint(0, 100)

        # re-normalize the randomized voting weights
        length_of_weights = math.sqrt(sum(map(lambda x: x ** 2, model.voting_weights.values())))

        for key, value in model.voting_weights.items():
            model.voting_weights[key] = value / length_of_weights

        yield model

        del model

    def test_prune_layer_by_vote_removal_removes_vote(self, model, nr_of_layers):
        """
        _prune_layer_by_vote_removal should remove the layer from model.voting_linear_layers
        """

        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)
            assert (str(index) not in model.voting_linear_layers.keys(),
                    "_prune_layer_by_vote_removal should remove the layer from model.voting_linear_layers")

    def test_prune_layer_by_vote_removal_does_not_remove_layer(self, model, nr_of_layers):
        """
        _prune_layer_by_vote_removal does not touch the layer in self.layers
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            prev = model.layers[index]
            model._prune_layer_by_vote_removal(index)
            assert model.layers[index] == prev, "_prune_layer_by_vote_removal does not touch the layer in self.layers"

    def test_prune_layer_by_vote_removal_removes_voting_weight(self, model, nr_of_layers):
        """
        _prune_layer_by_vote_removal should remove the layer from the voting weights
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)
            assert (index not in model.voting_weights.keys(),
                    "_prune_layer_by_vote_removal should remove the layer from the voting weights")


    def test_after_prune_layer_by_vote_removal_voting_weights_are_normalized(self, model, nr_of_layers, float_precision_tolerance):
        """
        _prune_layer_by_vote_removal should leave the voting weights normalized
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)
            assert (math.sqrt(sum(map(lambda x: x ** 2, model.voting_weights.values()))) - 1 <= float_precision_tolerance,
                "_prune_layer_by_vote_removal should leave the voting weights normalized")

    def test_after_prune_layer_by_vote_removal_voting_weights_are_correctly_normalized(self, model, nr_of_layers):
        """
        _prune_layer_by_vote_removal should not switch the voting weights
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            keys = [key for key in model.voting_weights.keys() if key != index]
            length_of_weights_without_key = (
                math.sqrt(
                    sum([value ** 2 for key, value in model.voting_weights.items() if key != index]))
            )
            weights_without_key_normalized = {
                key: value / length_of_weights_without_key
                for key, value in model.voting_weights.items()
                if key != index
            }

            model._prune_layer_by_vote_removal(index)
            assert (all([weights_without_key_normalized[key] - model.voting_weights[key] <= 10 ** -6 for key in keys]),
                "_prune_layer_by_vote_removal should not switch the voting weights")


    def test_prune_layer_by_vote_removal_does_not_break_forward_single_item(self, model, nr_of_layers, feature_count):
        """
        _prune_layer_by_vote_removal should not affect the functionality of the forward pass on single item batches
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(model, feature_count, msg="After performing _add_node: ")

    def test_prune_layer_by_vote_removal_does_not_break_backward_multiple_item(self, model, feature_count, nr_of_layers):
        """
        _prune_layer_by_vote_removal should not affect the functionality of the backward pass on multiple item batches
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)

        forward_tests = TestAutoDeepLearnerForward()
        batch_size = 1000
        forward_tests.test_forward_form_multiple_item_batch(model, feature_count, batch_size=batch_size,
                                                            msg=f"After performing _prune_layer_by_vote_removal on "
                                                                f"batch size {batch_size}: ")