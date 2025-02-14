from typing import Optional, Union

import torch
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.stream import Schema
from torch import nn
from torch.optim import Optimizer

from ADLClassifier.Resources.LearningRateProgressions import BaseLearningRateProgression
from ADLClassifier.BaseClassifier import ADLClassifier
from Model import AutoDeepLearner


def add_weight_correction_parameter_to_user_choices(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):

    class AddWeightCorrectionParameterToUserChoicesWrapper(adl_classifier):
        def __init__(self,
                     schema: Optional[Schema] = None,
                     random_seed: int = 1,
                     nn_model: Optional[AutoDeepLearner] = None,
                     optimizer: Optional[Optimizer] = None,
                     loss_fn=nn.CrossEntropyLoss(), device: str = "cpu",
                     lr: Union[float | BaseLearningRateProgression] = 1e-3,
                     drift_detector: BaseDriftDetector = ADWIN(delta=1e-5),
                     drift_criterion: str = "accuracy",
                     mci_threshold_for_layer_pruning: float = 10 ** -7,
                     layer_weight_learning_rate : float = 1e-3,
                     ):
            super().__init__(schema, random_seed, nn_model, optimizer, loss_fn, device, lr, drift_detector, drift_criterion,
                             mci_threshold_for_layer_pruning)
            self.layer_weight_learning_rate = layer_weight_learning_rate

        def __str__(self):
            return f"{super().__str__()}WithUserChosenWeightLR"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithUserChosenWeightLR"

        def _adjust_weights(self, true_label: torch.Tensor, step_size: float):
            super()._adjust_weights(true_label, self.layer_weight_learning_rate)

    AddWeightCorrectionParameterToUserChoicesWrapper.__name__ = f"{adl_classifier.__name__}WithUserChosenWeightLR"
    return AddWeightCorrectionParameterToUserChoicesWrapper