import pytest
import torch

from AutoDeepLearner import AutoDeepLearner


class TestAutoDeepLearnerForward:
    def test_forward_form_single_item_batch(self):
        model = AutoDeepLearner(5, 3)
        prediction = model(torch.rand(5))

        assert torch.is_tensor(prediction), 'prediction should be a tensor'
        assert prediction.size() == torch.tensor(1.0).size(), 'prediction should be of the shape of a scalar'

    def test_forward_form_multiple_item_batch(self):
        model = AutoDeepLearner(5, 3)
        prediction = model(torch.rand(9, 5))

        assert torch.is_tensor(prediction), 'prediction should be a tensor'
        assert prediction.size() == torch.tensor(1.0).size(), 'prediction should be of the shape of a scalar'
