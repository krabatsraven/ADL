import torch

from ADLClassifier.BaseClassifier import ADLClassifier


def train_only_winning_layer(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):
    class WithWinningLayerTrainingWrapper(adl_classifier):
        def __str__(self):
            return f"{super().__str__()}WithWinningLayerTraining"

        def _backpropagation(self, prediction: torch.Tensor, true_label: torch.Tensor):
            layers_to_disable = self.model.active_and_learning_layer_keys_wo_winning_layer().tolist()
            self.model._disable_layers_for_training(layers_to_disable)
            super()._backpropagation(prediction=prediction, true_label=true_label)
            self.model._enable_layers_for_training(layers_to_disable)

    return WithWinningLayerTrainingWrapper
