import random

import pytest

from Model import AutoDeepLearner
from tests.resources import random_initialize_model
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


class TestAutoDeepLearnerIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope='class')
    def iteration_count(self) -> int:
        return random.randint(1_000, 10_000)

    @pytest.fixture(scope='class', autouse=True)
    def model(self,
              feature_count: int,
              class_count: int,
              iteration_count: int
              ) -> AutoDeepLearner:

        model = AutoDeepLearner(nr_of_features=feature_count, nr_of_classes=class_count)

        model = random_initialize_model(model, iteration_count)

        yield model

        del model

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