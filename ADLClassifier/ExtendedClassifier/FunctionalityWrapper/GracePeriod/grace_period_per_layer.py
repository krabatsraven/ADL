import torch

from ADLClassifier.BaseClassifier import ADLClassifier


def grace_period_per_layer(adl_classifier: type(ADLClassifier), duration: int) -> type(ADLClassifier):
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
            super().__init__(*args, **kwargs)
            self.duration: int = duration
            self.time_since_last_change: torch.Tensor = torch.zeros(self.model.nr_of_active_layers, dtype=torch.int)
            self.model_changed_this_iteration: torch.Tensor = torch.zeros(self.model.nr_of_active_layers, dtype=torch.bool)

        def __str__(self):
            return f"{super().__str__()}WithGracePeriodPerLayerOf{self.duration}Instances"

        def train(self, instance):
            self.model_changed_this_iteration = torch.zeros(self.model.nr_of_active_layers, dtype=torch.bool)
            super().train(instance)
            self.time_since_last_change[~self.model_changed_this_iteration] += 1

        def _delete_layer(self, layer_index: int) -> None:
            if self.time_since_last_change[layer_index] > self.duration:
                super()._delete_layer(layer_index)
                self.time_since_last_change = torch.concat((self.time_since_last_change[:layer_index], self.time_since_last_change[layer_index + 1:]))
                self.model_changed_this_iteration = torch.concat((self.model_changed_this_iteration[:layer_index], self.model_changed_this_iteration[layer_index + 1:]))

        def _add_layer(self) -> None:
            if self.time_since_last_change[-1] > self.duration:
                super()._add_layer()
                self.model_changed_this_iteration = torch.concat((self.model_changed_this_iteration, torch.ones(1, dtype=torch.bool)))
                self.time_since_last_change = torch.concat((self.model_changed_this_iteration, torch.zeros(1, dtype=torch.int)))

        def _add_node(self, layer_index: int) -> None:
            if self.time_since_last_change[layer_index] > self.duration:
                super()._add_node(layer_index)
                self.model_changed_this_iteration[layer_index] = True
                self.time_since_last_change[layer_index] = 0

        def _delete_node(self, layer_index: int, node_index: int) -> None:
            if self.time_since_last_change[layer_index] > self.duration:
                super()._delete_node(layer_index, node_index)
                self.model_changed_this_iteration[layer_index] = True
                self.time_since_last_change[layer_index] = 0

    return GracePeriodPerLayerWrapper
