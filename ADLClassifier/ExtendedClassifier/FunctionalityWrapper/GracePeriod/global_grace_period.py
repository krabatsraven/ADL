from collections.abc import Callable
from typing import Dict, Any

import torch

from ADLClassifier.BaseClassifier import ADLClassifier


def global_grace_period(duration: int = 1000) -> Callable[[type(ADLClassifier)], type(ADLClassifier)]:
    return lambda adl_class: _global_grace_period(adl_classifier=adl_class, duration=duration)


def _global_grace_period(adl_classifier: type(ADLClassifier), duration: int) -> type(ADLClassifier):
    """
    Adds a grace period to the decorated/passed class of ADLClassifier
    for the grace period no changes will be applied.
    the grace period is measured in number of instances seen since the last change
    :param adl_classifier: the class of ADLClassifier that should get a grace period
    :param duration: the number of instances of data seen for which no change is to be applied
    :return: a class of ADLClassifier that has a global grace period added
    """

    class GlobalGracePeriodWrapper(adl_classifier):
        """
        ADLClassifier with a Global Grace Period
        """

        def __init__(self, *args, **kwargs):
            assert duration > 0, (f"grace period has to be a positive number, "
                                  f"if you want to disable the grace period consider using {adl_classifier} instead")
            super().__init__(*args, **kwargs)
            self.__duration: int = duration
            self.time_since_last_change: int = 0
            self.model_changed_this_iteration: bool = False

        def __str__(self):
            return f"{super().__str__()}WithGlobalGPOf{duration}Insts"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithGlobalGPOf{duration}Insts"

        def train(self, instance):
            self.model_changed_this_iteration = False
            super().train(instance)
            if not self.model_changed_this_iteration:
                self.time_since_last_change += 1

        def _backpropagation(self, prediction: torch.Tensor, true_label: torch.Tensor):
            if len(self.model.active_and_learning_layer_keys()) == 0:
                return
            super()._backpropagation(prediction, true_label)

        def _delete_layer(self, layer_index: int) -> bool:
            out = False
            if self.time_since_last_change > self.duration:
                out = super()._delete_layer(layer_index)
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0
            return out

        def _add_layer(self) -> bool:
            out = False
            if self.time_since_last_change > self.duration:
                out = super()._add_layer()
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0
            return out

        def _add_node(self, layer_index: int) -> bool:
            out = False
            if self.time_since_last_change > self.duration:
                out = super()._add_node(layer_index)
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0
            return out

        def _delete_node(self, layer_index: int, node_index: int) -> bool:
            out = False
            if self.time_since_last_change > self.duration:
                out = super()._delete_node(layer_index, node_index)
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0
            return out

        @property
        def duration(self) -> int:
            """
            The duration of the grace period
            :return: the nr of instances for which no changes are to be applied to the models layer since the last change
            """
            return self.__duration

        @property
        def state_dict(self) -> Dict[str, Any]:
            state_dict = super().state_dict
            state_dict['time_since_last_change'] = self.time_since_last_change
            state_dict['model_changed_this_iteration'] = self.model_changed_this_iteration
            state_dict['duration'] = self.duration
            return state_dict

        @state_dict.setter
        def state_dict(self, state_dict: Dict[str, Any]) -> None:
            adl_classifier.state_dict.__set__(self, state_dict)
            self.time_since_last_change = state_dict['time_since_last_change']
            self.model_changed_this_iteration = state_dict['model_changed_this_iteration']
            self.__duration = state_dict['duration']

    GlobalGracePeriodWrapper.__name__ = f"{adl_classifier.__name__}WithGlobalGPOf{duration}Insts"
    return GlobalGracePeriodWrapper
