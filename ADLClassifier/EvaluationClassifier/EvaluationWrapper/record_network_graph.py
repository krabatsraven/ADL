from io import BytesIO
from typing import Dict, Any

import torch
from capymoa._pickle import JPickler, JUnpickler
from capymoa.evaluation import ClassificationEvaluator

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
                "winning_layer": [],
                "learning_rate": []
            }
            # tracks only the cumulative results, window size is only the nr of times it saves those
            self.evaluator = ClassificationEvaluator(self.schema, window_size=1)

        def __str__(self):
            return f"{super().__str__()}WithGraphRecord"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithGraphRecord"

        def train(self, instance):
            if self.model is None:
                self.set_model(instance)

            self._test(instance)
            super().train(instance)

        def _test(self, instance):
            # save the nr of different layers for later evaluations
            self._update_record_of_model_shape()

            # test the performance of the model before training
            true_label = torch.tensor(instance.y_index, dtype=torch.long)
            true_label = torch.unsqueeze(true_label.to(self.device),0)
            prediction = self.predict(instance)
            self.evaluator.update(true_label.item(), prediction)

        def _update_record_of_model_shape(self):
            self.record_of_model_shape["nr_of_layers"].append(len(self.model.layers))
            self.record_of_model_shape["shape_of_hidden_layers"].append([tuple(list(h.weight.size()).__reversed__()) for h in self.model.layers])
            self.record_of_model_shape["active_layers"].append(self.model.active_layer_keys().numpy())
            self.record_of_model_shape["winning_layer"].append(self.model.get_winning_layer())
            self.record_of_model_shape['learning_rate'].append(self.learning_rate)

        @property
        def state_dict(self) -> Dict[str, Any]:
            state_dict = super().state_dict
            eval_file = BytesIO()
            JPickler(eval_file).dump(self.evaluator)
            eval_file.seek(0)
            state_dict["evaluator"] = eval_file
            state_dict['record_of_model_shape'] = {key: tuple(value) for key, value in self.record_of_model_shape.items()}
            return state_dict

        @state_dict.setter
        def state_dict(self, state_dict: Dict[str, Any]):
            adl_classifier.state_dict.__set__(self, state_dict)
            self.evaluator = JUnpickler(state_dict['evaluator']).load()
            state_dict['record_of_model_shape'] = {key: list(value) for key, value in self.record_of_model_shape.items()}

    return NetworkGraphRecorder
