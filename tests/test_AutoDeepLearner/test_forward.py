import pytest
import torch

from AutoDeepLearner import AutoDeepLearner


class TestAutoDeepLearnerForward:
    def test_forward_form_single_item_batch(self):
        feature_count, class_count = torch.randint(10_000, (2,))
        model = AutoDeepLearner(feature_count, class_count)
        prediction = model(torch.rand(feature_count))

        assert torch.is_tensor(prediction), 'prediction should be a tensor'
        assert prediction.size() == torch.tensor(1.0).size(), 'prediction should be of the shape of a scalar'

    def test_forward_form_multiple_item_batch(self):
        feature_count, class_count, batch_size = torch.randint(10_000, (3,))
        model = AutoDeepLearner(feature_count, class_count)
        prediction = model(torch.rand(batch_size, feature_count))

        assert torch.is_tensor(prediction), 'prediction should be a tensor'
        assert prediction.size() == torch.tensor(1.0).size(), 'prediction should be of the shape of a scalar'
