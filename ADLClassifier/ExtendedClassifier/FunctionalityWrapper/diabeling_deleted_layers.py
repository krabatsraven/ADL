from ADLClassifier.BaseClassifier import ADLClassifier


def disabeling_deleted_layers(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):
    """
    extends an existing ADLClassifier class 
    to disable the gradiant calculation for the corresponding hidden layer
    if an output layer is deleted
    :param adl_classifier: the class of ADL Classifier that should be extended
    :return: the extended ADLClassifier class
    """

    class DisabelingDeletedLayersWrapper(adl_classifier):
        """
        :arg class of ADLClassifier that sets requires_grad to False for hidden layers whose output layer has been deleted
        """

        def __str__(self):
            return f"{super().__str__()}WithDisabledDeletedLayers"

        def _delete_layer(self, layer_index: int) -> None:
            super()._delete_layer(layer_index)
            self.model.layers[layer_index].requires_grad_(False)

    return DisabelingDeletedLayersWrapper