from typing import Optional, Union, Dict, Any

import torch
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Schema
from torch import nn
from torch.optim import Optimizer

from ADLClassifier.Resources.LearningRateProgressions import BaseLearningRateProgression
from ADLClassifier.BaseClassifier import ADLClassifier
from Model import AutoDeepLearner

ADD_WEIGHT_CORRECTION_PARAMETER_NAME = "WithWeightLR"

def add_weight_correction_parameter_to_user_choices(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):

    class AddWeightCorrectionParameterToUserChoicesWrapper(adl_classifier):
        def __init__(self, *args, **kwargs):
            assert 'layer_weight_learning_rate' in kwargs.keys(), \
                    "Expected 'layer_weight_learning_rate' to be a provided as kwarg for decoupeling the lr from the weight correction parameters"
            self.layer_weight_learning_rate = kwargs.pop('layer_weight_learning_rate')

            super().__init__(*args, **kwargs)

        def __str__(self):
            return f"{super().__str__()}{ADD_WEIGHT_CORRECTION_PARAMETER_NAME}"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}{ADD_WEIGHT_CORRECTION_PARAMETER_NAME}"

        def _adjust_weights(self, true_label: torch.Tensor, step_size: float):
            super()._adjust_weights(true_label, self.layer_weight_learning_rate)

        @property
        def state_dict(self) -> Dict[str, Any]:
            state_dict = super().state_dict
            state_dict['layer_weight_learning_rate'] = self.layer_weight_learning_rate
            return state_dict

        @state_dict.setter
        def state_dict(self, state_dict: Dict[str, Any]) -> None:
            adl_classifier.state_dict.__set__(self, state_dict)
            self.layer_weight_learning_rate = state_dict['layer_weight_learning_rate']

    AddWeightCorrectionParameterToUserChoicesWrapper.__name__ = f"{adl_classifier.__name__}{ADD_WEIGHT_CORRECTION_PARAMETER_NAME}"
    return AddWeightCorrectionParameterToUserChoicesWrapper