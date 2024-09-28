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

    def test_forward_raises_exception_on_wrong_nr_of_features_on_one_datapoint(self, model: AutoDeepLearner, feature_count: int) -> None:
        wrong_feature_count = random.choice(list(range(feature_count)) + list(range(feature_count + 1, 10_000)))
        x = torch.rand(wrong_feature_count)

        with (
            pytest.raises(
                Exception,
                match=f"Given batch of data has {x.size()[0]} many features, expected where {model.input_size}"
            )):
            model.forward(x)

    def test_forward_raises_exception_on_wrong_nr_of_features_in_batches(self, model: AutoDeepLearner, feature_count: int, batch_size: int = 10_000) -> None:
        batch_size = random.randint(0, batch_size)
        wrong_feature_count = random.choice(list(range(feature_count)) + list(range(feature_count + 1, 10_000)))
        x = torch.rand(batch_size, wrong_feature_count)

        with (
            pytest.raises(
                Exception,
                match=f"Given batch of data has {x.size()[1]} many features, expected where {model.input_size}"
            )):
            model.forward(x)

    def test_forward_form_single_item_batch(self, model: AutoDeepLearner, feature_count: int, class_count: int, msg: str = ''):
        prediction = model(torch.rand(feature_count, requires_grad=True, dtype=torch.float))

        print(prediction)

        assert torch.is_tensor(prediction), f'{msg}prediction should be a tensor'
        assert prediction.size() == torch.ones(class_count).size(), f'{msg}prediction should be of the shape (1, nr of classes)'
        assert prediction.dtype == torch.float, f'{msg}prediction dtype should be float'
        assert torch.all(prediction >= 0), f'{msg}prediction should be bigger than zero as they represent class probabilities'
        assert torch.all(prediction >= 0), f'{msg}prediction should smaller than one as they represent class probabilities'

    def test_forward_form_multiple_item_batch(self, model: AutoDeepLearner, feature_count: int, class_count: int, batch_size: int = 10_000, msg: str = ''):
        batch_size = random.randint(0, batch_size)
        prediction = model(torch.rand(batch_size, feature_count))

        assert torch.is_tensor(prediction), f'{msg}prediction should be a tensor'
        assert prediction.size() == torch.ones(batch_size, class_count).size(), f'{msg}prediction should be of the shape (nr of batches, nr of classes)'
        assert prediction.dtype == torch.float, f'{msg}prediction dtype should be float'
        assert torch.all(prediction >= 0), f'{msg}prediction should be bigger than zero as they represent class probabilities'
        assert torch.all(prediction >= 0), f'{msg}prediction should smaller than one as they represent class probabilities'

