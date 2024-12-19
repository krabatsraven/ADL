import random
from typing import List

import numpy as np
import pytest
from numpy._typing import NDArray

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


def has_normalised_voting_weights(model_to_test: AutoDeepLearner) -> bool:
    """
    Checks if the models voting weights are normalized
    :param model_to_test: the AutoDeepLearner model to test
    :return: bool, true if the models voting weights are normalized
    """
    voting_weights_values_vector: NDArray[float] = model_to_test.get_voting_weight_values().numpy()
    norm_of_voting_weights: np.floating = np.linalg.norm(voting_weights_values_vector, ord=1)
    return norm_of_voting_weights - 1.0 <= 10 ** -6


class TestAutoDeepLearnerNormaliseVotingWeights:
    @pytest.fixture(scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def nr_of_layers(self) -> int:
        return 100

    @pytest.fixture(scope="class", autouse=True)
    def model(self, feature_count, class_count, nr_of_layers) -> AutoDeepLearner:
        model = AutoDeepLearner(feature_count, class_count)

        for idx in range(nr_of_layers):
            model._add_layer()

        yield model

        del model

    def test_static_method_returns_false_on_not_normalised_vector(self, model: AutoDeepLearner, nr_of_layers: int):
        key_choice = random.choice(model.active_layer_keys())
        model._AutoDeepLearner__set_voting_weight(int(key_choice), 2.0)
        assert not has_normalised_voting_weights(model), "The voting weights should not be normalised anymore"

    def test_static_method_returns_true_on_normalised_vector(self, model: AutoDeepLearner):
        for key in model.active_layer_keys():
            if key != 0:
                model._AutoDeepLearner__set_voting_weight(int(key), 0.0)
            else:
                model._AutoDeepLearner__set_voting_weight(int(key), 1.0)

        assert has_normalised_voting_weights(model), "The voting weights should be normalised"

    def test_normalise_works_on_already_normalised_voting_weights(self, model: AutoDeepLearner):
        assert has_normalised_voting_weights(model), \
            "Assuring that the initial models voting vector is normalised"
        model._normalise_voting_weights()
        assert has_normalised_voting_weights(model), \
            "The model should still possess a normalised voting vector after normalising a normalised vector"

    def test_normalise_works_on_not_normalised_vectors(self, model: AutoDeepLearner):
        for key in model.active_layer_keys():
            if key != 0:
                model._AutoDeepLearner__set_voting_weight(int(key), random.randint(0, 100))
            else:
                model._AutoDeepLearner__set_voting_weight(int(key), 2.0)

        assert not has_normalised_voting_weights(model), \
            "Assuring that the initial models voting vector is not normalised anymore"
        model._normalise_voting_weights()
        assert has_normalised_voting_weights(model), \
            "The model should possess a normalised voting vector after normalising"

    def test_normalise_raises_on_all_zeros_voting_weights_vector(self, model: AutoDeepLearner):
        for key in model.active_layer_keys():
            model._AutoDeepLearner__set_voting_weight(int(key), 0)

        with pytest.raises(Exception) as excinfo:
            model._normalise_voting_weights()

        assert str(excinfo.value) == "The voting weights vector has a length of zero and cannot be normalised"

    def test_normalise_does_not_break_forward_on_single_batch(self, model: AutoDeepLearner, feature_count: int, class_count: int):
        for key in model.active_layer_keys():
            if key != 0:
                model._AutoDeepLearner__set_voting_weight(int(key), random.randint(0, 100))
            else:
                model._AutoDeepLearner__set_voting_weight(int(key), 2.0)
        model._normalise_voting_weights()

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            class_count,
            msg="After performing _normalise_voting_weights: "
        )

    def test_normalise_does_not_break_forward_on_multiple_batch(self, model: AutoDeepLearner, feature_count: int, class_count: int):
        for key in model.active_layer_keys():
            if key != 0:
                model._AutoDeepLearner__set_voting_weight(int(key), random.randint(0, 100))
            else:
                model._AutoDeepLearner__set_voting_weight(int(key), 2.0)
        model._normalise_voting_weights()

        forward_tests = TestAutoDeepLearnerForward()
        batch_size = 1000
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            class_count,
            batch_size=batch_size,
            msg=f"After performing _normalise_voting_weights on a "
                f"batch size of {batch_size}: "
        )

