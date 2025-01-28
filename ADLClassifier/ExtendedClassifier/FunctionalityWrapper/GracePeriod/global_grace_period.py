from ADLClassifier.BaseClassifier import ADLClassifier


def global_grace_period(adl_classifier: type(ADLClassifier), duration: int) -> type(ADLClassifier):
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
            super().__init__(*args, **kwargs)
            self.duration: int = duration
            self.time_since_last_change: int = 0
            self.model_changed_this_iteration: bool = False

        def __str__(self):
            return f"{super().__str__()}WithGlobalGracePeriodOf{self.duration}Instances"

        def train(self, instance):
            self.model_changed_this_iteration = False
            super().train(instance)
            if not self.model_changed_this_iteration:
                self.time_since_last_change += 1

        def _delete_layer(self, layer_index: int) -> None:
            if self.time_since_last_change > self.duration:
                super()._delete_layer(layer_index)
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0

        def _add_layer(self) -> None:
            if self.time_since_last_change > self.duration:
                super()._add_layer()
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0

        def _add_node(self, layer_index: int) -> None:
            if self.time_since_last_change > self.duration:
                super()._add_node(layer_index)
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0

        def _delete_node(self, layer_index: int, node_index: int) -> None:
            if self.time_since_last_change > self.duration:
                super()._delete_node(layer_index, node_index)
                self.model_changed_this_iteration = True
                self.time_since_last_change = 0

    return GlobalGracePeriodWrapper
