import math
import random

import pytest

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

        for i in range(nr_of_layers - 1):
            model._add_layer()

            model.voting_weights[i + 1] = random.randint(0, 100)

        # re-normalize the randomized voting weights
        model._normalise_voting_weights()

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

    def test_prune_layer_by_vote_removal_does_not_break_forward_single_item(self, model, nr_of_layers, class_count, feature_count):
        """
        _prune_layer_by_vote_removal should not affect the functionality of the forward pass on single item batches
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            class_count,
            msg="After performing _add_node: "
        )

    def test_prune_layer_by_vote_removal_does_not_break_backward_multiple_item(self, model, feature_count, class_count, nr_of_layers):
        """
        _prune_layer_by_vote_removal should not affect the functionality of the backward pass on multiple item batches
        """
        for index in random.sample(range(nr_of_layers), random.randint(1, nr_of_layers)):
            model._prune_layer_by_vote_removal(index)

        forward_tests = TestAutoDeepLearnerForward()
        batch_size = 1000
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            class_count,
            batch_size=batch_size,
            msg=f"After performing _prune_layer_by_vote_removal on "
                                                                f"batch size {batch_size}: "
        )

    def test_prune_layer_by_vote_removal_raises_on_negative_index(self, model):
        layer_index = -1
        error_str = f"cannot remove the layer with the index {layer_index}, as it is not in the range [0, amount of layers in model]"
        with pytest.raises(Exception) as exec_info:
            model._prune_layer_by_vote_removal(layer_index)

        assert str(exec_info.value) == error_str, "negative indices should raise an exception"

    def test_prune_layer_by_vote_removal_raises_on_to_big_index(self, model):
        layer_index = len(model.layers)
        error_str = f"cannot remove the layer with the index {layer_index}, as it is not in the range [0, amount of layers in model]"
        with pytest.raises(Exception) as exec_info:
            model._prune_layer_by_vote_removal(layer_index)

        assert str(exec_info.value) == error_str, ("indices bigger than or equal to "
                                                   "the length of the list of layers should raise an exception")

    def test_prune_layer_layer_by_vote_removal_raises_on_no_voting_linear_layer(self, model, nr_of_layers):
        layer_index = random.randint(0, nr_of_layers - 1)
        model.voting_linear_layers.pop(str(layer_index))
        error_str = (f"cannot remove the layer with the index {layer_index}, "
                     f"as it is not a layer that will projected onto a vote")
        with pytest.raises(Exception) as exec_info:
            model._prune_layer_by_vote_removal(layer_index)

        assert str(exec_info.value) == error_str, \
            "a layer index without a voting linear layer should raise an exception"

    def test_prune_layer_layer_by_vote_removal_raises_on_no_voting_weight(self, model, nr_of_layers):
        layer_index = random.randint(0, nr_of_layers - 1)
        model.voting_weights.pop(layer_index)
        error_str = (f"cannot remove the layer with the index {layer_index}, "
                     f"as it is not a layer that can vote because it has no voting weight")
        with pytest.raises(Exception) as exec_info:
            model._prune_layer_by_vote_removal(layer_index)

        assert str(exec_info.value) == error_str, \
            "a layer index without a voting weight should raise an exception"

    def test_prune_layer_layer_by_vote_removal_raises_on_removal_of_last_non_zero_voting_weight(self, model, nr_of_layers):
        layer_index = random.randint(0, nr_of_layers - 1)
        error_str = (f"cannot remove the layer with the index {layer_index}, "
                     f"as it is the last layer with a non zero voting weight")

        for idx in range(nr_of_layers + 1):
            if idx != layer_index:
                model.voting_weights[idx] = 0.0

        with pytest.raises(Exception) as exec_info:
            model._prune_layer_by_vote_removal(layer_index)

        assert str(exec_info.value) == error_str, \
            "the last layer with a non zero voting weight should raise an exception if attempted to be removed"