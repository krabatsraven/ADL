import pytest
import random

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


def add_layer_test(model, iter, msg):
    nr_of_layers_before_adding = len(iter)
    model._add_layer()
    assert len(iter) == nr_of_layers_before_adding + 1, msg


class TestAutoDeepLearnerAddLayer:

    @pytest.fixture(scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(autouse=True, scope="class")
    def nr_of_layers(self) -> int:
        return 100

    @pytest.fixture(autouse=True)
    def model(self, feature_count, class_count) -> AutoDeepLearner:
        model = AutoDeepLearner(feature_count, class_count)

        yield model

        del model

    def test_add_layer_adds_layer(self, model, nr_of_layers):

        for i in range(nr_of_layers):
            add_layer_test(model, model.layers, "__add_layer should result in model having a single layer more than "
                                                "before")

        assert len(
            model.layers) == nr_of_layers + 1, f"model should have {nr_of_layers + 1} layers after adding a {nr_of_layers} layers"

    def test_new_layers_vote(self, model, nr_of_layers):

        for i in range(nr_of_layers):
            add_layer_test(model, model.voting_linear_layers, "__add_layer should result in model having a single "
                                                              "layer more that votes than before")

        assert len(
            model.voting_linear_layers) == nr_of_layers + 1, f"model should have {nr_of_layers + 1} voting layers after adding a {nr_of_layers} layers"

    def test_new_layers_voting_weight(self, model, nr_of_layers):

        for i in range(nr_of_layers):
            add_layer_test(model, model.voting_linear_layers, "__add_layer should result in model having a single "
                                                              "voting weight more that votes than before")

        assert len(
            model.voting_weights) == nr_of_layers + 1, f"model should have {nr_of_layers + 1} voting weights after adding a {nr_of_layers} layers"

        assert sum(model.voting_weights.values()) == 1, "models voting weights should be normalised"

    def test_add_layer_should_still_not_break_forward(self, model, feature_count, nr_of_layers):
        """
        After performing _add_layer the functionality of forward should still be intact
        """

        for _ in range(nr_of_layers):
            model._add_layer()

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(model, feature_count, msg="After performing _add_layer: ")

    def test_add_layer_should_still_not_break_forward_multiple(self, model, feature_count, nr_of_layers):
        """
        After performing _add_layer the functionality of forward should still be intact
        """

        for _ in range(nr_of_layers):
            model._add_layer()

        forward_tests = TestAutoDeepLearnerForward()
        # setting batch size too big will break the memory
        forward_tests.test_forward_form_multiple_item_batch(model, feature_count, batch_size=1000,
                                                            msg="After performing _add_layer: ")
