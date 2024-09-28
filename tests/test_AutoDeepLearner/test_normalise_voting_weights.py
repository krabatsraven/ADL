import random
from typing import List

import numpy as np
import pytest
import torch
from numpy._typing import NDArray

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward

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

    @pytest.fixture(scope="class")
    def max_amount_of_nodes(self):
        return 100

    @pytest.fixture(scope="class")
    def nr_of_nodes_added_in_layer(self, nr_of_layers, max_amount_of_nodes) -> List[int]:
        return [random.randint(1, max_amount_of_nodes) for _ in range(nr_of_layers)]

    @pytest.fixture(autouse=True)
    def model(self, feature_count, class_count, nr_of_layers, nr_of_nodes_added_in_layer) -> AutoDeepLearner:
        model = AutoDeepLearner(feature_count, class_count)

        for idx in range(nr_of_layers):
            model._add_layer()
            for _ in range(nr_of_nodes_added_in_layer[idx]):
                model._add_node(idx)

        yield model

        del model

    @staticmethod
    def has_normalised_voting_weights(model_to_test: AutoDeepLearner) -> bool:
        voting_weights_values_vector: NDArray[float] = np.fromiter(model_to_test.voting_weights.values(), dtype=float)
        norm_of_voting_weights: np.floating = np.linalg.norm(voting_weights_values_vector)
        return norm_of_voting_weights == 1.0

    def test_static_method_returns_false_on_not_normalised_vector(self):
        pass

    def test_static_method_returns_true_on_normalised_vector(self):
        pass

    def test_normalise_works_on_already_normalised_voting_weights(self, model: AutoDeepLearner):
        assert self.has_normalised_voting_weights(model), "Assuring that the initial model has a normalised voting vector"
        model._normalise_voting_weights()
        assert self.has_normalised_voting_weights(model), "The model should still have a normalised voting vector after normalising a normalised vector"

    def test_normalise_works_on_not_normalised_vectors(self, model: AutoDeepLearner):
        pass

    def test_normalise_raises_on_all_zeros_voting_weights_vector(self, model: AutoDeepLearner):
        pass

    def test_normalise_does_not_break_forward_on_single_batch(self, model: AutoDeepLearner):
        pass

    def test_normalise_does_not_break_forward_on_multiple_batch(self, model: AutoDeepLearner):
        pass
