from typing import Callable, Dict, Any

import torch

from ADLClassifier.BaseClassifier import ADLClassifier


GRACE_PERIOD_PER_LAYER_NAME = 'WithGPPerLayerOf'
GRACE_PERIOD_PER_LAYER_NAME_LAMBDA = lambda duration: f"{GRACE_PERIOD_PER_LAYER_NAME}{duration}Insts"

def grace_period_per_layer(duration: int = 1000) -> Callable[[type(ADLClassifier)], type(ADLClassifier)]:
    return lambda adl_class: _grace_period_per_layer(adl_classifier=adl_class, duration=duration)


def _grace_period_per_layer(adl_classifier: type(ADLClassifier), duration: int) -> type(ADLClassifier):
    """
    Adds a grace period to the decorated/passed class of ADLClassifier
    for the grace period of a layer no changes to the layer will be applied
    as well as in case of the last layer no additional layer will be added.
    the grace period is measured in number of instances seen since the last change
    :param adl_classifier: the class of ADLClassifier that should get a grace period
    :param duration: the number of instances of data seen for which no change is to be applied to a layer
    :return: a class of ADLClassifier that has a grace period per layer added
    """
    class GracePeriodPerLayerWrapper(adl_classifier):
        """
        ADLClassifier with a Grace Period per Layer
        """

        def __init__(self, *args, **kwargs):
            assert duration > 0, (f"grace period has to be a positive number, "
                                  f"if you want to disable the grace period consider using {adl_classifier} instead")
            super().__init__(*args, **kwargs)
            self.__duration: int = duration
            self.time_since_last_change: torch.Tensor = torch.zeros(self.model.nr_of_active_layers, dtype=torch.int)
            self.model_changed_this_iteration: torch.Tensor = torch.zeros(self.model.nr_of_active_layers, dtype=torch.bool)

        def __str__(self):
            return f"{super().__str__()}{GRACE_PERIOD_PER_LAYER_NAME_LAMBDA(duration)}"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}{GRACE_PERIOD_PER_LAYER_NAME_LAMBDA(duration)}"

        def train(self, instance):
            self.model_changed_this_iteration = torch.zeros(self.model.nr_of_active_layers, dtype=torch.bool)
            super().train(instance)
            self.time_since_last_change[~self.model_changed_this_iteration] += 1

        def _backpropagation(self, prediction: torch.Tensor, true_label: torch.Tensor):
            if len(self.model.active_and_learning_layer_keys()) == 0:
                return
            super()._backpropagation(prediction, true_label)

        def _delete_layer(self, layer_index: int) -> bool:
            out = False
            output_layer_index = self._output_layer_index(layer_index)
            if self.time_since_last_change[output_layer_index] > self.__duration:
                out = super()._delete_layer(layer_index)
                self.time_since_last_change = torch.concat((self.time_since_last_change[:output_layer_index], self.time_since_last_change[output_layer_index + 1:]))
                self.model_changed_this_iteration = torch.concat((self.model_changed_this_iteration[:output_layer_index], self.model_changed_this_iteration[output_layer_index + 1:]))
            return out

        def _add_layer(self) -> bool:
            out = False
            if self.time_since_last_change[-1] > self.__duration:
                out = super()._add_layer()
                self.model_changed_this_iteration = torch.concat((self.model_changed_this_iteration, torch.ones(1, dtype=torch.bool)))
                self.time_since_last_change = torch.concat((self.time_since_last_change, torch.zeros(1, dtype=torch.int)))
            return out

        def _add_node(self, layer_index: int) -> bool:
            out = False
            output_layer_index = self._output_layer_index(layer_index)
            if self.time_since_last_change[output_layer_index] > self.__duration:
                out = super()._add_node(layer_index)
                self.model_changed_this_iteration[output_layer_index] = True
                self.time_since_last_change[output_layer_index] = 0
            return out

        def _delete_node(self, layer_index: int, node_index: int) -> bool:
            out = False
            output_layer_index = self._output_layer_index(layer_index)
            if self.time_since_last_change[output_layer_index] > self.__duration:
                out = super()._delete_node(layer_index, node_index)
                self.model_changed_this_iteration[output_layer_index] = True
                self.time_since_last_change[output_layer_index] = 0
            return out

        def _output_layer_index(self, layer_index: int) -> int:
            return (self.model.active_layer_keys() == layer_index).nonzero().item()

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

    GracePeriodPerLayerWrapper.__name__ = f"{adl_classifier.__name__}{GRACE_PERIOD_PER_LAYER_NAME_LAMBDA(duration)}"
    return GracePeriodPerLayerWrapper
