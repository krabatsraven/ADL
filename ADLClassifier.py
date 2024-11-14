from typing import Union, Optional

import numpy as np
import torch
from capymoa.base import Classifier
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationEvaluator, RegressionEvaluator
from capymoa.stream import Schema
from torch import nn
from torch.optim import Optimizer

from AutoDeepLearner import AutoDeepLearner


class ADLClassifier(Classifier):
    def __init__(
            self,
            schema: Optional[Schema] = None,
            random_seed: int = 1,
            nn_model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            loss_fn=nn.CrossEntropyLoss(),
            device: str = ("cpu"),
            lr: float = 1e-3,
            evaluator: ClassificationEvaluator = ClassificationEvaluator(),
            drift_detector: BaseDriftDetector = ADWIN(delta=0.001),
            drift_criterion: str = "accuracy"
    ):

        super().__init__(schema, random_seed)
        self.model: Optional[AutoDeepLearner] = None
        self.optimizer: Optional[Optimizer] = None
        self.loss_function = loss_fn
        self.learning_rate: float = lr
        self.device: str = device

        torch.manual_seed(random_seed)

        if nn_model is None:
            self.set_model(None)
        else:
            self.model = nn_model.to(device)
        if optimizer is None and self.model is not None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        self.evaluator: ClassificationEvaluator = evaluator
        self.drift_detector: BaseDriftDetector = drift_detector
        self.__drift_criterion_switch = drift_criterion

        self.drift_warning_data: Optional[torch.Tensor] = None
        self.drift_warning_label: Optional[torch.Tensor] = None

    def __str__(self):
        return str(self.model)

    def CLI_help(self):
        return str('schema=None, random_seed=1, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=("cpu"), lr=1e-3 evaluator=ClassificationEvaluator(), drift_detector=ADWIN(delta=0.001), drift_criterion="accuracy"')

    def set_model(self, instance):
        if self.schema is None:
            moa_instance = instance.java_instance.getData()

            self.model = AutoDeepLearner(
                nr_of_features = moa_instance.get_num_attributes(),
                nr_of_classes = moa_instance.get_num_classes()
            ).to(self.device)
        elif instance is not None:
            self.model = AutoDeepLearner(
                nr_of_features = self.schema.get_num_attributes(),
                nr_of_classes = self.schema.get_num_classes()
            ).to(self.device)

    def train(self, instance):
        if self.model is None:
            self.set_model(instance)

        X = torch.tensor(instance.x, dtype=torch.float32)
        y = torch.tensor(instance.y_index, dtype=torch.long)
        # set the device and add a dimension to the tensor
        X, y = torch.unsqueeze(X.to(self.device), 0), torch.unsqueeze(y.to(self.device),0)

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_function(pred, y)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        self.evaluator.update(y.item(), torch.argmax(pred).item())
        self.drift_detector.add_element(self.__drift_criterion(y, pred))

        self._adjust_weights(true_label=y, step_size=self.learning_rate)
        # todo: # 71
        # todo: # 72
        self._high_lvl_learning(true_label=y, prediction=pred, data=X)
        # low lvl

        self.optimizer.zero_grad()

    def predict(self, instance):
        return np.argmax(self.predict_proba(instance))

    def predict_proba(self, instance):
        if self.model is None:
            self.set_model(instance)
        X = torch.unsqueeze(torch.tensor(instance.x, dtype=torch.float32).to(self.device), 0)
        # turn off gradient collection
        with torch.no_grad():
            pred = np.asarray(self.model(X).numpy(), dtype=np.double)
        return pred

    # -----------------
    # internal functions for training

    # todo: new name for _adjust_weight()
    def _adjust_weights(self, true_label: torch.Tensor, step_size: float):
        # find the indices of the results that where correctly predicted
        correctly_predicted_layers_indices = np.where(torch.argmax(self.model.layer_results, dim = 1) == true_label)
        correctly_predicted_layers_mask = np.zeros(self.model.layer_result_keys.shape, dtype=bool)
        correctly_predicted_layers_mask[correctly_predicted_layers_indices] = True

        weight_correction_factor_values = self.model.get_weight_correction_factor_values()
        step_size_tensor = torch.tensor(step_size, dtype=torch.float)

        # if layer predicted correctly increase weight correction factor p^(l) by step_size "zeta"
        # p^(l) = p^(l) + step_size
        weight_correction_factor_values[correctly_predicted_layers_mask] += step_size_tensor
        # if layer predicted erroneous decrease weight correction factor p^(l) by step size
        # p^(l) = p^(l) - step_size
        weight_correction_factor_values[~correctly_predicted_layers_mask] -= step_size_tensor

        # assure that epsilon <= p^(l) <= 1
        weight_correction_factor_values = torch.where(
            weight_correction_factor_values > self.model.upper_weigth_correction_factor_boarder,
            self.model.upper_weigth_correction_factor_boarder,
            weight_correction_factor_values
        )
        weight_correction_factor_values = torch.where(
            weight_correction_factor_values < self.model.lower_weigth_correction_factor_boarder,
            self.model.lower_weigth_correction_factor_boarder,
            weight_correction_factor_values
        )

        # create mapping of values and update parameter dict of network
        weight_correction_factor_items = {
            key: value.item()
            for key, value in zip(
                self.model.weight_correction_factor.keys(),
                weight_correction_factor_values
            )
        }
        self.model.weight_correction_factor.update(weight_correction_factor_items)

        # adjust voting weight of layer l:
        voting_weight_values = self.model.get_voting_weight_values()

        # if layer l was correct increase beta:
        # beta^(l) = (1 + p^(l)) * beta^(l)
        voting_weight_values[correctly_predicted_layers_mask] *= (1 + weight_correction_factor_values[correctly_predicted_layers_mask])

        # if layer l was incorrect decrease beta:
        # beta^(l) = p^(l) * beta^(l)
        voting_weight_values[~correctly_predicted_layers_mask] *= weight_correction_factor_values[~correctly_predicted_layers_mask]

        # while assuring that 0 <= beta^(l) <= 1
        voting_weight_values = torch.where(
            voting_weight_values > self.model.upper_voting_weight_boarder,
            self.model.upper_voting_weight_boarder,
            voting_weight_values
        )
        voting_weight_values = torch.where(
            voting_weight_values < self.model.lower_voting_weigth_boarder,
            self.model.lower_voting_weigth_boarder,
            voting_weight_values
        )

        # create mapping of values and update parameter dict of network
        voting_weight_items = {
            key: value.item()
            for key, value in zip(
                self.model.voting_weights.keys(),
                voting_weight_values
            )
        }
        self.model.voting_weights.update(voting_weight_items)

    def _high_lvl_learning(self, true_label: torch.Tensor, prediction: torch.Tensor, data: torch.Tensor):
        # prune highly correlated layers:
        # ------------------------------

        # find correlated layers:
        # compare the predictions of the layer pairwise
        correlations = torch.corrcoef(self.model.layer_results)
        # prune them

        # grow the hidden layers to accommodate concept drift when it happens:
        # -------------------------------------------------------------------
        if self.drift_detector.detected_change():
            # stack saved data if there is any onto the current instance to train with both
            if self.drift_warning_data is not None:
                data = torch.stack((self.drift_warning_data, data))
                true_label = torch.stack((self.drift_warning_label, true_label))

            # add layer
            self.model._add_layer()

            # train the layer:
            # todo: question #69
                # freeze the parameters not new:
                    # originaly they create a new network with only one layer and train the weights there
                    # can we just delete the gradients of all weights not in the new layer?
                # train by gradient
            pred = self.model.forward(data, exclude_layer_indicies_in_training=list(range(len(self.model.layers) - 1)))
            loss = self.loss_function(pred, true_label)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            # low level training
            # todo: comment in low level if implemented
            # self._low_lvl_learning()
            pass

        elif self.drift_detector.detected_warning():
            # store instance
            # todo: question: #70
            self.drift_warning_data = data
            self.drift_warning_label = true_label

        else:
            # stable phase means deletion of buffered warning instances
            self.drift_warning_data = None
            self.drift_warning_label = None

    def _low_lvl_learning(self):
        raise NotImplementedError

    def __drift_criterion(self, true_label: torch.Tensor, prediction: torch.Tensor) -> float:
        match self.__drift_criterion_switch:
            case "accuracy":
                # use accuracy to univariant detect concept drift
                return self.evaluator.accuracy()

            case "loss":
                # use the loss to univariant detect concept drift
                return self.loss_function(true_label, prediction)
