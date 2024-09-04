import random

import pytest
import torch

from AutoDeepLearner import AutoDeepLearner


class TestAutoDeepLearnerForward:

    @pytest.fixture(autouse=True, scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(autouse=True)
    def model(self, feature_count, class_count) -> AutoDeepLearner:
        return AutoDeepLearner(feature_count, class_count)

    def test_forward_form_single_item_batch(self, model: AutoDeepLearner, feature_count: int, msg: str = ''):
        prediction = model(torch.rand(feature_count))

        assert torch.is_tensor(prediction), f'{msg}prediction should be a tensor'
        assert prediction.size() == torch.tensor(1.0).size(), f'{msg}prediction should be of the shape of a scalar'

    def test_forward_form_multiple_item_batch(self, model: AutoDeepLearner, feature_count: int, batch_size: int = 10_000, msg: str = ''):
        batch_size = random.randint(0, batch_size)
        prediction = model(torch.rand(batch_size, feature_count))

        assert torch.is_tensor(prediction), f'{msg}prediction should be a tensor'
        # assert prediction.size() == torch.tensor(1.0).size(), f'{msg}prediction should be of the shape of a scalar'
