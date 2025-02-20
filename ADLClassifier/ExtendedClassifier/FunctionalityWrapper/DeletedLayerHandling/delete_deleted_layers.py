from ADLClassifier import ADLClassifier


def delete_deleted_layers(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):
    """
    extends an existing ADLClassifier class 
    to disable the gradiant calculation for the corresponding hidden layer
    if an output layer is deleted
    :param adl_classifier: the class of ADL Classifier that should be extended
    :return: the extended ADLClassifier class
    """

    class DeleteDeletedLayersWrapper(adl_classifier):
        """
        :arg class of ADLClassifier that sets requires_grad to False for hidden layers whose output layer has been deleted
        """

        def __str__(self):
            return f"{super().__str__()}WithDeleteDeletedLayers"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithDeleteDeletedLayers"

        def _delete_layer(self, layer_index: int) -> bool:
            # not exactly the same output, as we remove a sigmoid function between both layers in the forward stack
            if super()._delete_layer(layer_index):
                self.model.delete_hidden_layer(layer_index)
                return True
            else:
                return False

    DeleteDeletedLayersWrapper.__name__ = f"{adl_classifier.__name__}WithDeleteDeletedLayers"
    return DeleteDeletedLayersWrapper
