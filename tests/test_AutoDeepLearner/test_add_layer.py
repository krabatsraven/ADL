import math

import pytest
import random

from Model import AutoDeepLearner
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
            add_layer_test(
                model,
                model.layers,
                "__add_layer should result in model having a single layer more than before"
            )

        assert len(model.layers) == nr_of_layers + 1,\
            f"model should have {nr_of_layers + 1} layers after adding a {nr_of_layers} layers"

    def test_new_layers_vote(self, model, nr_of_layers):

        for i in range(nr_of_layers):
            add_layer_test(
                model,
                model.voting_linear_layers,
                "__add_layer should result in model having a single layer more that votes than before"
            )

        assert len(model.voting_linear_layers) == nr_of_layers + 1,\
            f"model should have {nr_of_layers + 1} voting layers after adding a {nr_of_layers} layers"

    def test_new_layers_voting_weight(self, model, nr_of_layers):

        for i in range(nr_of_layers):
            add_layer_test(
                model,
                model.voting_weights,
                "__add_layer should result in model having a single voting weight more that votes than before"
            )

        assert len(model.voting_weights) == nr_of_layers + 1,\
            f"model should have {nr_of_layers + 1} voting weights after adding a {nr_of_layers} layers"

        # the following test is an open question
        # whether the models voting weights should be manually normalised after the add step
        # assert has_normalised_voting_weights(model), "models voting weights should be normalised"

    def test_new_layers_voting_weight_correction_factor(self, model, nr_of_layers):
        for i in range(nr_of_layers):
            add_layer_test(
                model,
                model.weight_correction_factor,
                "__add_layer should result in the model having exactly a single voting weight correction factor "
                "more that votes than before"
            )

            assert model.weight_correction_factor_with_index_exists(i + 1), \
                f"the key {i + 1} should be added after adding the {i + 1}-te layer"

            assert model.get_weight_correction_factor(i + 1) == model.weight_correction_factor_initialization_value, \
                f"the {i + 1}-te layer should be initialised with the correct factor"

        assert len(model.weight_correction_factor) == nr_of_layers + 1,\
            (f"model should have {nr_of_layers + 1} voting weight correction factor"
             f" after adding a {nr_of_layers} layers")

    def test_add_layer_should_still_not_break_forward(self, model, feature_count, class_count, nr_of_layers):
        """
        After performing _add_layer the functionality of forward should still be intact
        """

        for _ in range(nr_of_layers):
            model._add_layer()

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            class_count,
            msg="After performing _add_layer: "
        )

    def test_add_layer_should_still_not_break_forward_multiple(self, model, feature_count, class_count, nr_of_layers):
        """
        After performing _add_layer the functionality of forward should still be intact
        """

        for _ in range(nr_of_layers):
            model._add_layer()

        forward_tests = TestAutoDeepLearnerForward()
        # setting batch size too big will break the memory
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            class_count,
            batch_size=1000,
            msg="After performing _add_layer: "
        )
