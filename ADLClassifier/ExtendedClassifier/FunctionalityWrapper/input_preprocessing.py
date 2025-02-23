from typing import Optional, Union

import numpy as np
import torch
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Schema
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.optim import Optimizer

from ADLClassifier.Resources.LearningRateProgressions import BaseLearningRateProgression
from ADLClassifier.BaseClassifier import ADLClassifier
from Model import AutoDeepLearner


def input_preprocessing(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):

    class PrecocessingInputWrapper(adl_classifier):
        """
        Wrapper that adds normalization and one hot encoding to instances before passing them to the learning algorithmn
        """
        
        def __init__(self, schema: Optional[Schema] = None, random_seed: int = 1,
                     nn_model: Optional[AutoDeepLearner] = None, optimizer: Optional[Optimizer] = None,
                     loss_fn=nn.CrossEntropyLoss(), device: str = "cpu",
                     lr: Union[float | BaseLearningRateProgression] = 1e-3,
                     drift_detector: BaseDriftDetector = ADWIN(delta=1e-5), drift_criterion: str = "accuracy",
                     mci_threshold_for_layer_pruning: float = 10 ** -7):
            super().__init__(schema, random_seed, nn_model, optimizer, loss_fn, device, lr, drift_detector, drift_criterion,
                             mci_threshold_for_layer_pruning)
    
            nominal_indicies = [i for i in range(self.schema.get_num_attributes()) if self.schema.get_moa_header().attribute(i).isNominal()]
            numerical_indicies = [i for i in range(self.schema.get_num_attributes()) if self.schema.get_moa_header().attribute(i).isNumeric()]
            nominal_values = [torch.arange(len(self.schema.get_moa_header().attribute(i).getAttributeValues()), dtype=torch.float32) for i in nominal_indicies]
            self.input_transformer = make_column_transformer(
                (OneHotEncoder(categories=nominal_values), nominal_indicies),
                (StreamingStandardScaler(), numerical_indicies),
                remainder='passthrough',
                sparse_threshold=0)

        def __str__(self):
            return f"{super().__str__()}WithInputProcessing"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithInputProcessing"

        def _preprocess_instance(self, instance):
            if len(instance.x.shape) < 2:
                # adding a dimension to data to ensure shape of [batch_size(=1), nr_of_features]
                X = instance.x[np.newaxis, ...]
            else:
                X = instance.x
            if self.nr_of_instances_seen == 0:
                self.input_transformer.fit(X)
            return torch.from_numpy(self.input_transformer.transform(X)).to(device=self.device, dtype=torch.float)

    return PrecocessingInputWrapper


class StreamingStandardScaler(StandardScaler):

    def fit(self, X, y=None, sample_weight=None):
        return self.partial_fit(X, y, sample_weight=sample_weight)

    def transform(self, X, copy=None):
        return super().transform(X, copy)