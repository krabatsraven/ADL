import pytest
import torch

from AutoDeepLearner import AutoDeepLearner


class TestAutoDeepLearnerAddLayer:

    def test_add_layer_adds_layer(self):
        feature_count, class_count = torch.randint(10_000, (2,))
        model = AutoDeepLearner(feature_count, class_count)

        for i in range(100):
            nr_of_layers_before_adding = len(model.layers)
            model._add_layer()
            assert len(model.layers) == nr_of_layers_before_adding + 1, (
                "__add_layer should result in model having a single layer more than before")

        assert len(model.layers) == 101, "model should have 101 layers after adding a hundred layers"

    def test_new_layers_vote(self):
        feature_count, class_count = torch.randint(10_000, (2,))
        model = AutoDeepLearner(feature_count, class_count)

        for i in range(100):
            nr_of_layers_before_adding = len(model.layers)
            model._add_layer()
            assert len(model.voting_linear_layers) == nr_of_layers_before_adding + 1, (
                "__add_layer should result in model having a single layer more that votes than before")

        assert len(
            model.voting_linear_layers) == 101, "model should have 101 voting layers after adding a hundred layers"

    def test_new_layers_voting_weight(self):
        feature_count, class_count = torch.randint(10_000, (2,))
        model = AutoDeepLearner(feature_count, class_count)

        for i in range(100):
            nr_of_layers_before_adding = len(model.layers)
            model._add_layer()
            assert len(model.voting_weights) == nr_of_layers_before_adding + 1, (
                "__add_layer should result in model having a single voting weight more that votes than before")

        assert len(model.voting_weights) == 101, "model should have 101 voting weights after adding a hundred layers"

        assert sum(model.voting_weights.values()) == 1, "models voting weights should be normalised"
