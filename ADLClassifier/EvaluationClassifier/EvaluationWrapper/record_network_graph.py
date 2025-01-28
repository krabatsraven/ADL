import torch

from ADLClassifier import ADLClassifier


def record_network_graph(adl_classifier: type(ADLClassifier)):

    class NetworkGraphRecorder(adl_classifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # keep track of the shape of the model for each seen instance for later evaluation
            self.record_of_model_shape = {
                "nr_of_layers": [],
                "shape_of_hidden_layers": [],
                "active_layers": [],
                "winning_layer": []
            }

        def __str__(self):
            return f"{super().__str__()}WithGraphRecord"

        def train(self, instance):
            if self.model is None:
                self.set_model(instance)

            self._update_record_of_model_shape()
            super().train(instance)

        def _update_record_of_model_shape(self):
            self.record_of_model_shape["nr_of_layers"].append(len(self.model.layers))
            self.record_of_model_shape["shape_of_hidden_layers"].append([tuple(list(h.weight.size()).__reversed__()) for h in self.model.layers])
            self.record_of_model_shape["active_layers"].append(self.model.active_layer_keys().numpy())
            self.record_of_model_shape["winning_layer"].append(self.model.get_winning_layer())

        def __drift_criterion(self, true_label: torch.Tensor, prediction: torch.Tensor) -> float:
            # for evaluation it is necessary to track the statistical features, even if we use another criterion fro the drift
            self.evaluator.update(true_label.item(), torch.argmax(prediction).item())
            match self.__drift_criterion_switch:
                case "accuracy":
                    # use accuracy to univariant detect concept drift
                    return self.evaluator.accuracy()

                case "loss":
                    # use the loss to univariant detect concept drift
                    return self.loss_function(true_label, prediction)

    return NetworkGraphRecorder
