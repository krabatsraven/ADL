from functools import reduce
from io import BytesIO
from pickle import Pickler, Unpickler
from typing import Optional, Union, Any, Dict

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

        def __init__(self, schema: Schema, random_seed: int = 1,
                     nn_model: Optional[AutoDeepLearner] = None, optimizer: Optional[Optimizer] = None,
                     loss_fn=nn.CrossEntropyLoss(), device: str = "cpu",
                     lr: Union[float | BaseLearningRateProgression] = 1e-3,
                     drift_detector: BaseDriftDetector = ADWIN(delta=1e-5), drift_criterion: str = "accuracy",
                     mci_threshold_for_layer_pruning: float = 10 ** -7):
            self.schema = schema

            transformers = [
                elem
                for elem in
                [
                    (OneHotEncoder(categories=[torch.arange(len(self.schema.get_moa_header().attribute(i).getAttributeValues()), dtype=torch.float32)]), [i])
                    if self.schema.get_moa_header().attribute(i).isNominal()
                    else (StreamingStandardScaler(), [i])
                    if self.schema.get_moa_header().attribute(i).isNumeric()
                    else None
                    for i in range(self.schema.get_num_attributes())
                ]
                if elem is not None
            ]
            self.input_transformer = make_column_transformer(
                *transformers,
                remainder='passthrough',
                sparse_threshold=0)
            super().__init__(schema, random_seed, nn_model, optimizer, loss_fn, device, lr, drift_detector, drift_criterion,
                             mci_threshold_for_layer_pruning)

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
                self.nr_of_instances_seen += 1
                self.set_model(instance)
            else:
                self.nr_of_instances_seen += 1
            return torch.from_numpy(self.input_transformer.transform(X)).to(device=self.device, dtype=torch.float)

        @property
        def state_dict(self) -> Dict[str, Any]:
            state_dict = super().state_dict
            transformer_file = BytesIO()
            Pickler(transformer_file).dump(self.input_transformer)
            transformer_file.seek(0)
            state_dict['input_transformer'] = transformer_file
            return state_dict

        @state_dict.setter
        def state_dict(self, state_dict: Dict[str, Any]) -> None:
            adl_classifier.state_dict.__set__(self, state_dict)
            self.input_transformer = Unpickler(state_dict['input_transformer']).load()


    return PrecocessingInputWrapper


class StreamingStandardScaler(StandardScaler):

    def fit(self, X, y=None, sample_weight=None):
        # pass without doing anything async make_column_transformer() calls partial_fit everytime
        return self

    def transform(self, X, copy=None):
        self.partial_fit(X)
        return super().transform(X, copy)