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

INPUT_PREPROCESSING_NAME = 'WithInput'

def input_preprocessing(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):

    class PrecocessingInputWrapper(adl_classifier):
        """
        Wrapper that adds normalization and one hot encoding to instances before passing them to the learning algorithmn
        """

        def __init__(self, *args, **kwargs):
            if not args:
                assert 'schema' in kwargs.keys(), "Expected 'schema' to be a argument for input processing"
                self.schema = kwargs.get('schema')
            else:
                self.schema = args[0]

            super().__init__(*args, **kwargs)

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

        def __str__(self):
            return f"{super().__str__()}{INPUT_PREPROCESSING_NAME}"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}{INPUT_PREPROCESSING_NAME}"

        def _preprocess_instance(self, instance):
            if len(instance.x.shape) < 2:
                # adding a dimension to data to ensure shape of [batch_size(=1), nr_of_features]
                X = instance.x[np.newaxis, ...]
            else:
                X = instance.x
            if self.nr_of_instances_seen == 0:
                self.input_transformer.fit(X)
                self.nr_of_instances_seen = 1
                self.set_model(instance)
                self.nr_of_instances_seen = 0
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

    PrecocessingInputWrapper.__name__ = f"{adl_classifier.__name__}{INPUT_PREPROCESSING_NAME}"
    return PrecocessingInputWrapper


class StreamingStandardScaler(StandardScaler):

    def fit(self, X, y=None, sample_weight=None):
        # pass without doing anything async make_column_transformer() calls partial_fit everytime
        return self

    def transform(self, X, copy=None):
        self.partial_fit(X)
        return super().transform(X, copy)