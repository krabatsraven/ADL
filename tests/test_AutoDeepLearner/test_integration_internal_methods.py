import random

import pytest

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


class TestAutoDeepLearnerIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope='class', autouse=True)
    def model(self, feature_count, class_count) -> AutoDeepLearner:
        model = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)

        for _ in range(1000):
            dice = random.choice(range(1, 5))

            match dice:
                case 1:
                    model._add_layer()

                    # the add_layer function initialises the voting weight with 0
                    # not all voting weight can be zero: solution here: randomly assign a value between zero and one
                    last_added_layer_idx = len(model.layers) - 1
                    model.voting_weights[last_added_layer_idx] = random.uniform(0, 1)

                    # and normalise the voting weights
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

        return model

    def test_single_batch_integration(self, model, feature_count, class_count):

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            class_count,
            msg="Single Batch on a complex randomly grown model: "
        )

    def test_multi_batch_integration(self, model, feature_count, class_count):
        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            class_count,
            batch_size=1000,
            msg="Multiple Batch on a complex randomly grown model: "
        )