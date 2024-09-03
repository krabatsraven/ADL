import pytest
import torch

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


def model_setup() -> AutoDeepLearner:
    feature_count, class_count = torch.randint(10_000, (2,))
    return AutoDeepLearner(feature_count, class_count)

def add_layer_test(model, iter, msg):
    nr_of_layers_before_adding = len(iter)
    model._add_layer()
    assert len(iter) == nr_of_layers_before_adding + 1, msg


class TestAutoDeepLearnerAddLayer:

    def test_add_layer_adds_layer(self):
        model = model_setup()

        for i in range(100):
            add_layer_test(model, model.layers, "__add_layer should result in model having a single layer more than "
                                                "before")

        assert len(model.layers) == 101, "model should have 101 layers after adding a hundred layers"

    def test_new_layers_vote(self):
        model = model_setup()

        for i in range(100):
            add_layer_test(model, model.voting_linear_layers, "__add_layer should result in model having a single "
                                                              "layer more that votes than before")

        assert len(
            model.voting_linear_layers) == 101, "model should have 101 voting layers after adding a hundred layers"

    def test_new_layers_voting_weight(self):
        model = model_setup()

        for i in range(100):
            add_layer_test(model, model.voting_linear_layers, "__add_layer should result in model having a single "
                                                              "voting weight more that votes than before")

        assert len(model.voting_weights) == 101, "model should have 101 voting weights after adding a hundred layers"

        assert sum(model.voting_weights.values()) == 1, "models voting weights should be normalised"

    def test_add_layer_should_still_not_break_forward(self):
        """
        After performing _add_layer the functionality of forward should still be intact
        """

        model = model_setup()
        forward_tests = TestAutoDeepLearnerForward()

        for layer_idx in range(100):
            model._add_layer()

        forward_tests.test_forward_form_single_item_batch()
        forward_tests.test_forward_form_multiple_item_batch()
