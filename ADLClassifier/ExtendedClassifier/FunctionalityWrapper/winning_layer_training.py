import torch

from ADLClassifier.BaseClassifier import ADLClassifier


def winning_layer_training(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):
    """
    Changes the training/backpropagation of the decorated/passed class of ADLClassifier to only change the winning layer
    the winning layer is the layer with the highest voting weight
    after the weights are changed according to the correctness of the prediction on the current instance
    :param adl_classifier: the class of ADLClassifier that should be changed
    :return: a class of ADLClassifier that only trains the winning layer
    """
    class WithWinningLayerTrainingWrapper(adl_classifier):
        """
        A class of ADLClassifier that only trains the winning layer
        """

        def __str__(self):
            return f"{super().__str__()}WithWinningLayerTraining"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithWinningLayerTraining"

        def _backpropagation(self, prediction: torch.Tensor, true_label: torch.Tensor):
            layers_to_disable = self.model.active_and_learning_layer_keys_wo_winning_layer().tolist()
            self.model._disable_layers_for_training(layers_to_disable)
            super()._backpropagation(prediction=prediction, true_label=true_label)
            self.model._enable_layers_for_training(layers_to_disable)

    WithWinningLayerTrainingWrapper.__name__ = f"WinningLayer{adl_classifier.__name__}"
    return WithWinningLayerTrainingWrapper
