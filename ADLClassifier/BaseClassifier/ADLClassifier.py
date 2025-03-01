from io import BytesIO
from typing import Optional, Tuple, List, Union, Any, Dict

import numpy as np
import torch
from capymoa._pickle import JPickler, JUnpickler
from capymoa.base import Classifier
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationEvaluator
from capymoa.stream import Schema
from torch import nn
from torch.optim import Optimizer

from ADLClassifier.Resources import BaseLearningRateProgression
from Model import AutoDeepLearner


class ADLClassifier(Classifier):
    def __init__(
            self,
            schema: Optional[Schema] = None,
            random_seed: int = 1,
            nn_model: Optional[AutoDeepLearner] = None,
            optimizer: Optional[Optimizer] = None,
            loss_fn=nn.CrossEntropyLoss(),
            device: str = "cpu",
            lr: Union[float | BaseLearningRateProgression] = 1e-3,
            drift_detector: BaseDriftDetector = ADWIN(delta=1e-5),
            drift_criterion: str = "accuracy",
            mci_threshold_for_layer_pruning: float = 10**-7
    ):

        super().__init__(schema, random_seed)
        self.random_seed = random_seed
        self.model: Optional[AutoDeepLearner] = None
        self.optimizer: Optional[Optimizer] = None
        self.loss_function = loss_fn
        self.__learning_rate: Optional[Union[float | BaseLearningRateProgression]] = None
        self.learning_rate_progression: Optional[bool] = None
        self.device: str = device
        self._nr_of_instances_seen = 0

        # data and label which triggered a drift warning in the past
        self.drift_warning_data: Optional[torch.Tensor] = None
        self.drift_warning_label: Optional[torch.Tensor] = None

        # user chosen threshold for the mci value that guards the layer pruning
        self.mci_threshold_for_layer_pruning = mci_threshold_for_layer_pruning


        # values that are tracking the bias and variance of the model to inform node growing/pruning decisions
        #  ---------------------------------------------------------------------------------------------------
        self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance: torch.Tensor = torch.tensor(0, dtype=torch.int)

        self.mean_of_bias_squared: Optional[torch.Tensor] = None
        self.sum_of_bias_squared_residuals_squared: Optional[torch.Tensor] = None
        self.standard_deviation_of_bias_squared: Optional[torch.Tensor] = None

        self.mean_of_variance_squared_squared: Optional[torch.Tensor] = None
        self.sum_of_variance_squared_residuals_squared: Optional[torch.Tensor] = None
        self.standard_deviation_of_variance_squared_squared: Optional[torch.Tensor] = None

        self.minimum_of_mean_of_bias_squared: Optional[torch.Tensor] = None
        self.minimum_of_standard_deviation_of_bias_squared: Optional[torch.Tensor] = None
        self.minimum_of_mean_of_variance_squared_squared: Optional[torch.Tensor] = None
        self.minimum_of_standard_deviation_of_variance_squared_squared: Optional[torch.Tensor] = None


        self.learning_rate = lr
        torch.manual_seed(random_seed)

        if nn_model is None:
            self.set_model(None)
        else:
            self.model = nn_model.to(device)
        if optimizer is None and self.model is not None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

        self.adl_evaluator = ClassificationEvaluator(self.schema, window_size=1)

        self.drift_detector: BaseDriftDetector = drift_detector
        self.drift_criterion_switch = drift_criterion

        # instantiate variables to recursively calculate the covariance of output nodes to each other:
        # ---------------------------
        self.__init_cov_variables__()

    def __init_cov_variables__(self) -> None:
        # to later calculate the mci for layer removal
        # n_x_m = amount of output_layers * amount of nodes per output_layers (amount of classes)
        n_x_m = self.model.nr_of_active_layers * self.model.output_size
        # the matrix of residual sums
        self.sum_of_residual_output_probabilities: torch.Tensor = torch.zeros((n_x_m, n_x_m))
        # the row wise mean of all reshaped layer_results: \vec µ:
        self.mean_of_output_probabilities: torch.Tensor = torch.zeros(n_x_m, 1)
        # nr of instances seen in each layer
        self.nr_of_instances_seen_for_cov = torch.zeros(self.model.nr_of_active_layers, dtype=torch.int)

    def __str__(self):
        return "ADLClassifier"

    @classmethod
    def name(cls) -> str:
        return "ADLClassifier"

    def CLI_help(self):
        return str('schema=None, random_seed=1, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=("cpu"), lr=1e-3 evaluator=ClassificationEvaluator(), drift_detector=ADWIN(delta=0.001), drift_criterion="accuracy"')

    def set_model(self, instance):
        if instance is not None:
            nr_of_attributes = self._preprocess_instance(instance).shape[-1]

            self.model: AutoDeepLearner = AutoDeepLearner(
                nr_of_features = nr_of_attributes,
                nr_of_classes = instance.schema.get_num_classes()
            ).to(self.device)
        elif self.schema is not None:
            self.model: AutoDeepLearner = AutoDeepLearner(
                nr_of_features = self.schema.get_num_attributes(),
                nr_of_classes = self.schema.get_num_classes()
            ).to(self.device)


    def train(self, instance):
            if self.model is None:
                self.set_model(instance)
    
            self._train(instance)

    def predict(self, instance):
        return self.predict_proba(instance).argmax()

    def predict_proba(self, instance):
        if self.model is None:
            self.set_model(instance)
        X = self._preprocess_instance(instance)
        # turn off gradient collection
        with torch.no_grad():
            pred = np.asarray(self.model(X).numpy(), dtype=np.double)
        return pred

    # -----------------
    # internal functions for training
    def _train(self, instance):
        X, y = self._preprocess_trainings_instance(instance)
        # add data to tracked_statistical variables:
        self.model.add_data_to_statistical_variables(X)

        # Compute prediction error
        pred = self.model(X)

        self._adjust_weights(true_label=y, step_size=self.learning_rate)

        # Backpropagation
        self._backpropagation(prediction=pred, true_label=y)

        self.drift_detector.add_element(self._drift_criterion(y, pred))

        # todo: # 71
        self._high_lvl_learning(true_label=y, data=X)
        self._low_lvl_learning(true_label=y)
        self.optimizer.zero_grad()

    def _backpropagation(self, prediction, true_label):
        self.__backpropagation(prediction, true_label)

    def __backpropagation(self, prediction: torch.Tensor, true_label: torch.Tensor):
        loss = self.loss_function(prediction, true_label)
        loss.backward()

        if self.learning_rate_progression:
            self._reset_learning_rate()

        self.optimizer.step()

    def _preprocess_trainings_instance(self, instance):
        X = self._preprocess_instance(instance)
        y = torch.tensor(instance.y_index, dtype=torch.long)
        y = torch.unsqueeze(y.to(self.device),0)
        return X, y

    def _preprocess_instance(self, instance):
        self.nr_of_instances_seen += 1
        X = torch.tensor(instance.x, dtype=torch.float32)
        # set the device and add a dimension to the tensor
        return torch.unsqueeze(X.to(self.device), 0)

    # todo: new name for _adjust_weight()
    def _adjust_weights(self, true_label: torch.Tensor, step_size: float):
        # forward has to be executed beforehand, else adjust weight cannot determine:
        # find the indices of the results that where correctly predicted
        correctly_predicted_layers_mask_on_batch = torch.argmax(self.model.layer_results, dim = 2) == true_label

        weight_correction_factor_values = self.model.get_weight_correction_factor_values()
        step_size_tensor = torch.tensor(step_size, dtype=torch.float)

        for corrects_on_data_point in correctly_predicted_layers_mask_on_batch:
            # if layer predicted correctly increase weight correction factor p^(l) by step_size "zeta"
            # p^(l) = p^(l) + step_size
            if corrects_on_data_point.shape != weight_correction_factor_values.shape:
                print(weight_correction_factor_values)
                print(self.model.weight_correction_factor)
            weight_correction_factor_values[corrects_on_data_point] += step_size_tensor
            # if layer predicted erroneous decrease weight correction factor p^(l) by step size
            # p^(l) = p^(l) - step_size
            weight_correction_factor_values[~corrects_on_data_point] -= step_size_tensor

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
            voting_weight_values[corrects_on_data_point] *= (1 + weight_correction_factor_values[corrects_on_data_point])

            # if layer l was incorrect decrease beta:
            # beta^(l) = p^(l) * beta^(l)
            voting_weight_values[~corrects_on_data_point] *= weight_correction_factor_values[~corrects_on_data_point]

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

    def _high_lvl_learning(self, true_label: torch.Tensor, data: torch.Tensor):
        self.__high_level_learning_pruning_hidden_layers()
        self.__high_level_learning_grow_new_hidden_layers(data, true_label)

    def _get_correlated_pairs_of_output_layers(self) -> List[Tuple[int, int]]:
        # find correlated layers:
        # compare the predictions of the layer pairwise
        # layer_results are the stacked result_i, where result_i stems from output layer i
        # meaning the shape is (nr_of_layers, nr_of_classes)
        n = self.model.active_layer_keys().size(dim=0)
        m = self.model.output_size

        for row in self.model.layer_results:
            self._update_covariance_with_one_row(row)

        all_covariances_matrix = self._covariance_of_output_nodes()
        variances = torch.diag(all_covariances_matrix)
        mci_values = -1 * torch.ones((n, n, m))

        # # Calculate MCI for each pair of layers
        # # current idea: reshape cov to n X n x m, calculate with vectors of length m
        # # get combinations through a view instead of the 2 outer for loops?
        for layer_i_idx in range(n):
            for layer_j_idx in range(layer_i_idx + 1, n):
                for class_o_idx in range(m):
                    i = layer_i_idx*m + class_o_idx
                    j = layer_j_idx*m + class_o_idx
                    # Get auto-covariances (variances) and cross-covariance
                    variance_i = variances[i]
                    variance_j = variances[j]
                    cov_ij = all_covariances_matrix[i, j]

                    # Calculate Pearson correlation
                    pearson_corr = cov_ij / torch.sqrt(variance_i * variance_j)

                    # Calculate MCI using the formula
                    mci = 0.5 * (variance_i + variance_j) - torch.sqrt((variance_i + variance_j)**2 - 4 * variance_i * variance_j * (1 - pearson_corr**2))

                    # Store the MCI in array for layer i and layer j
                    mci_values[layer_i_idx, layer_j_idx, class_o_idx] = torch.absolute(mci)

        # find the correlated pairs by comparing the max mci for each pairing against a user chosen threshold
        mci_max_values = torch.max(mci_values, dim=2).values
        # the pair (0, 1) and (1, 0) should only appear once
        # later we prioritize to delete layer j before layer i if i < j
        # the maximal mci for a layer pair has to be smaller than the users threshold
        correlated_pairs = ((torch.logical_and(0 <= mci_max_values, mci_max_values < self.mci_threshold_for_layer_pruning))
                            .nonzero()
                            .sort()[0]
                            .unique(dim=0, sorted=True, return_counts=False, return_inverse=False)
                            .flip(dims=(0,))
                            )

        return [(self.model.active_layer_keys()[i].item(), self.model.active_layer_keys()[j].item()) for i, j in correlated_pairs]

    def __high_level_learning_pruning_hidden_layers(self):
        # prune highly correlated layers:
        # ------------------------------

        # find correlated layers
        correlated_pairs = self._get_correlated_pairs_of_output_layers()

        # prune them
        for index_of_layer_i, index_of_layer_j in correlated_pairs:
            # todo: issue #80
            if self.model.output_layer_with_index_exists(index_of_layer_i) and self.model.output_layer_with_index_exists(index_of_layer_j):
                # if both layers still exist:
                if self.model.get_voting_weight(index_of_layer_i) < self.model.get_voting_weight(index_of_layer_j):
                    # layer_i has the smaller voting weight and gets pruned
                    self._delete_layer(index_of_layer_i)
                else:
                    # voting_weight(j) <= voting_weight(i) => prune j
                    # as i <= j is guaranteed by beforehand sorting
                    # we prune j if voting_weight(i) == voting_weight(j)
                    self._delete_layer(index_of_layer_j)

    def __high_level_learning_grow_new_hidden_layers(self, data: torch.Tensor, true_label: torch.Tensor):
        # grow the hidden layers to accommodate concept drift when it happens:
        # -------------------------------------------------------------------
        if self.drift_detector.detected_change():
            # stack saved data if there is any onto the current instance to train with both
            if self.drift_warning_data is not None:
                data = torch.stack((self.drift_warning_data, data))
                true_label = torch.stack((self.drift_warning_label, true_label))

            # add layer
            active_layers_before_adding = self.model.active_and_learning_layer_keys().tolist()
            if self._add_layer():
                # train the layer:
                # todo: question #69
                # freeze the parameters not new:
                # originally they create a new network with only one layer and train the weights there
                # can we just delete the gradients of all weights not in the new layer?
                # train by gradient
                # disable all active layers that were not just added:

                self.model._disable_layers_for_training(active_layers_before_adding)
                pred = self.model.forward(data)
                self.__backpropagation(pred, true_label)
                # reactivate all but the newest layer (the newest should already be active):
                self.model._enable_layers_for_training(active_layers_before_adding)
                # low level training
            self._low_lvl_learning(true_label=true_label)

        elif self.drift_detector.detected_warning():
            # store instance
            # todo: question: #70
            self.drift_warning_data = data
            self.drift_warning_label = true_label

        else:
            # stable phase means deletion of buffered warning instances
            self.drift_warning_data = None
            self.drift_warning_label = None

    def _low_lvl_learning(self, true_label: torch.Tensor):
        # get true prediction
        true_prediction = torch.zeros(self.model.output_size)
        true_prediction[true_label] = 1
        winning_layer = self.model.get_winning_layer()

        expected_value_of_winning_layer, expected_value_winning_layer_squared, idx_of_least_contributing_winning_layer_node = self.model.get_expected_value_expected_squared_value_and_idx_of_least_contributing_node_for_layer(winning_layer)

        # update mean and standard deviation of bias^2 and variance^2 with expected and expected squared value:
        # new bias^2:
        bias_of_winning_layer = (true_prediction - expected_value_of_winning_layer)
        bias_of_winning_layer_squared = (bias_of_winning_layer.matmul(bias_of_winning_layer))
        # new variance^2:
        variance_of_winning_layer = expected_value_of_winning_layer ** 2 - expected_value_winning_layer_squared
        variance_of_winning_layer_squared = variance_of_winning_layer.matmul(variance_of_winning_layer)
        # update mean, standard deviation and minima of biases and variances so far:
        self._update_mu_and_s_of_bias_and_var(bias_of_winning_layer_squared, variance_of_winning_layer_squared)

        # \kappa
        kappa = 1.3 * torch.exp(-bias_of_winning_layer_squared) + 0.7
        # if criterion 1: µ_bias + s_bias >= min_µ_bias + \kappa * min_s_bias
        if (self.mean_of_bias_squared + self.standard_deviation_of_bias_squared 
                >= self.minimum_of_mean_of_bias_squared + kappa * self.minimum_of_standard_deviation_of_bias_squared):
            # add node to winning layer
            self._add_node(winning_layer)
            # return: we don't add and shrink in the same turn
            return

        # \chi
        xi = 1.3 * torch.exp(-variance_of_winning_layer_squared) + 0.7
        # if criterion 2:  µ_var + s_var >= min_µ_var + 2 * \chi * min_s_var
        if (self.mean_of_variance_squared_squared + self.standard_deviation_of_variance_squared_squared 
                >= self.minimum_of_mean_of_variance_squared_squared + 2 * xi *self.minimum_of_standard_deviation_of_variance_squared_squared):
            # remove the least contributing node from the winning layer
            self._delete_node(winning_layer, idx_of_least_contributing_winning_layer_node)

            # if the node is deleted the minima are reset:
            self.minimum_of_mean_of_bias_squared = self.mean_of_bias_squared
            self.minimum_of_standard_deviation_of_bias_squared = self.standard_deviation_of_bias_squared
            self.minimum_of_mean_of_variance_squared_squared = self.mean_of_variance_squared_squared
            self.minimum_of_standard_deviation_of_variance_squared = self.standard_deviation_of_variance_squared_squared

    def _update_mu_and_s_of_bias_and_var(self, bias_squared: torch.Tensor, variance_squared: torch.Tensor) -> None:
        if self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance == 0:
            self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance += 1

            self.mean_of_bias_squared = bias_squared
            self.standard_deviation_of_bias_squared = torch.tensor(0, dtype=torch.float, requires_grad=False)
            self.sum_of_bias_squared_residuals_squared = torch.tensor(0, dtype=torch.float, requires_grad=False)

            self.mean_of_variance_squared_squared = variance_squared
            self.standard_deviation_of_variance_squared_squared = torch.tensor(0, dtype=torch.float, requires_grad=False)
            self.sum_of_variance_squared_residuals_squared = torch.tensor(0, dtype=torch.float, requires_grad=False)

            self.minimum_of_mean_of_bias_squared = self.mean_of_bias_squared
            self.minimum_of_standard_deviation_of_bias_squared = self.standard_deviation_of_bias_squared
            self.minimum_of_mean_of_variance_squared_squared = self.mean_of_variance_squared_squared
            self.minimum_of_standard_deviation_of_variance_squared_squared = self.standard_deviation_of_variance_squared_squared

            return

        self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance += 1

        # \bar x_n = \bar x_{n-1} + \frac{x_n - \bar x_{n-1}}{n}
        new_mean_of_bias_squared = (self.mean_of_bias_squared 
                                    + (
                                            (bias_squared - self.mean_of_bias_squared) 
                                            / self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance
                                    )
                                    )
        # M_{2,n} = M_{2, n-1} + (x_n - \bar x_{n})(x_n - \bar x_{n-1})
        self.sum_of_bias_squared_residuals_squared += (
                (bias_squared - new_mean_of_bias_squared)
                * (bias_squared - self.mean_of_bias_squared)
        )
        self.mean_of_bias_squared = new_mean_of_bias_squared
        # s = \sqrt{\frac{M_{2,n}}{n-1}}
        self.standard_deviation_of_bias_squared = torch.sqrt(
            self.sum_of_bias_squared_residuals_squared
            / (self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance - 1)
        )

        # \bar x_n = \bar x_{n-1} + \frac{x_n - \bar x_{n-1}}{n}
        new_mean_of_variance_squared = (self.mean_of_variance_squared_squared 
                                        + (
                                                (variance_squared - self.mean_of_variance_squared_squared)
                                                / self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance
                                        )
                                        )

        # M_{2,n} = M_{2, n-1} + (x_n - \bar x_{n})(x_n - \bar x_{n-1})
        self.sum_of_variance_squared_residuals_squared += (
            (variance_squared - new_mean_of_variance_squared)
            * (variance_squared - self.mean_of_variance_squared_squared)
        )
        self.mean_of_variance_squared_squared = new_mean_of_variance_squared
        # s = \sqrt{\frac{M_{2,n}}{n-1}}
        self.standard_deviation_of_variance_squared_squared = torch.sqrt(
                self.sum_of_variance_squared_residuals_squared 
                / (self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance - 1)
        )

        self.minimum_of_mean_of_bias_squared = torch.min(
            self.minimum_of_mean_of_bias_squared,
            self.mean_of_bias_squared
        )
        self.minimum_of_standard_deviation_of_bias_squared = torch.min(
            self.minimum_of_standard_deviation_of_bias_squared,
            self.standard_deviation_of_bias_squared
        )
        self.minimum_of_mean_of_variance_squared_squared = torch.min(
            self.minimum_of_mean_of_variance_squared_squared,
            self.mean_of_variance_squared_squared
        )
        self.minimum_of_standard_deviation_of_variance_squared_squared = torch.min(
            self.minimum_of_standard_deviation_of_variance_squared_squared,
            self.standard_deviation_of_variance_squared_squared
        )

    def _drift_criterion(self, true_label: torch.Tensor, prediction: torch.Tensor) -> float:
        match self.drift_criterion_switch:
            case "accuracy":
                # use accuracy to univariant detect concept drift
                self.adl_evaluator.update(true_label.item(), torch.argmax(prediction).item())
                return self.adl_evaluator.accuracy()

            case "loss":
                # use the loss to univariant detect concept drift
                return self.loss_function(true_label, prediction)

    def _update_covariance_with_one_row(self, layer_result: torch.Tensor):
        assert layer_result.shape == (self.model.nr_of_active_layers, self.model.output_size), \
            (f"updating the covariance with more than one result leads to numerical instability:"
             f" shape of layer_result: {layer_result.shape}"
             f" expected shape: {(self.model.nr_of_active_layers, self.model.output_size)}")
        # covariance matrix: cov = ((layer_x1_class_y1_prob)_{i=x1+y1}, (layer_x2_class_y2_prob)_{j=x2+y2}))_{i, j}
        # 0 <= x1, x2 < nr_of_active_layers; 0 <= y1, y2 < nr_of_classes
        # layer_result = (layer_i_class_j_prob)_{ij}
        # reshape layer_result to vector of shape 1 x (nr_of_active_layers*nr_of_classes): \vec l

        # n += 1
        self.nr_of_instances_seen_for_cov += 1
        # dx = x - mean_x (of n-1 instances)
        residual_w_mean_x_n_minus_1 = (
                (
                    layer_result
                    .reshape(
                        (
                            self.model.nr_of_active_layers*self.model.output_size,
                            1
                        )
                    )
                )
                - self.mean_of_output_probabilities
        )

        # \bar x_n = \bar {x_{n-1}} + \frac{x_n - \bar{x_{n-1}}}{n} = mean_x + dx / n
        self.mean_of_output_probabilities = (
                self.mean_of_output_probabilities
                +
                (
                        residual_w_mean_x_n_minus_1
                        /
                        (
                            self.nr_of_instances_seen_for_cov
                            .unsqueeze(1)
                            .expand(-1, self.model.output_size)
                            .reshape((-1, 1))
                        )
                )
        )
        # meany += (y - meany) / n
        # C += dx * (y - meany)
        # C_n   := sum_{i=1}^n(x_i - \bar{x_n})(y_i - \bar{y_n})
        #       = C_{n-1} + (x_n - \bar{x_n})(y_n - \bar{y_{n - 1}})
        #       = C_{n-1} + (x_{n} - \bar{x_{n - 1}})(y_n - \bar{y_{n}})
        self.sum_of_residual_output_probabilities = (
                self.sum_of_residual_output_probabilities
                +
                (
                    residual_w_mean_x_n_minus_1
                    .mul(
                        (
                                layer_result.reshape(1, self.model.nr_of_active_layers * self.model.output_size)
                                -
                                self.mean_of_output_probabilities.reshape(1, -1)
                        )
                    )
                )
        )

    def _covariance_of_output_nodes(self):
        # C_n   := sum_{i=1}^n(x_i - \bar{x_n})(y_i - \bar{y_n})
        #       = C_{n-1} + (x_n - \bar{x_n})(y_n - \bar{y_{n - 1}})
        #       = C_{n-1} + (x_{n} - \bar{x_{n - 1}})(y_n - \bar{y_{n}})
        #
        # bechel correction if n = 1:
        # that means x_1 = \bar x => C_1 = 0 => cov = 0 without bechel correction
        #
        # sample_cov(x,y) = \frac{C_n}{n - 1}
        # the matrix of all sample_cov(x,y)_{i,j} = \frac{(C_n)_{i,j}}{n - 1}
        nr_of_instances_stretched = (
            self.nr_of_instances_seen_for_cov
            .unsqueeze(1)
            .expand(-1, self.model.output_size)
            .reshape((-1, 1))
        )
        return (
            (self.sum_of_residual_output_probabilities / (nr_of_instances_stretched - 1))
            .where(nr_of_instances_stretched > 1, 0)
        )

    def _add_layer_to_covariance_matrix(self):
        # new layers are added: add columns and rows of zeros to the end of cov matrix for each class:
        self.sum_of_residual_output_probabilities = torch.cat(
            (
                torch.cat(
                    (
                        self.sum_of_residual_output_probabilities,
                        torch.zeros(self.model.output_size, self.sum_of_residual_output_probabilities.shape[1])
                    ),
                    dim=0
                ),
                torch.zeros(self.sum_of_residual_output_probabilities.shape[0] + self.model.output_size, self.model.output_size)
            ),
            dim=1
        )
        # and to instances/mean seen:
        self.nr_of_instances_seen_for_cov = torch.cat((self.nr_of_instances_seen_for_cov, torch.zeros(1)))
        self.mean_of_output_probabilities = torch.cat((self.mean_of_output_probabilities, torch.zeros(self.model.output_size, 1)))

    def _remove_layer_from_covariance_matrix(self, layer_idx: int):
        assert self.model.output_layer_with_index_exists(layer_idx), "we can only remove a layer from the covariance that still has an output layer"
        # if layer x is removed: remove rows i and columns j between x * nr_of_classes <= i, j < (x + 1) * nr_of_classes
        # the index x of the layer to remove is the position of the layer_id in relation to the other active layers
        idx_to_remove_relative_to_matrix = torch.nonzero(self.model.active_layer_keys() == layer_idx).item()

        first_index_to_remove_form_matrix = idx_to_remove_relative_to_matrix * self.model.output_size
        first_index_to_not_remove_anymore_form_matrix = (idx_to_remove_relative_to_matrix + 1) * self.model.output_size
        # remove the row
        self.sum_of_residual_output_probabilities = torch.cat(
            (
                self.sum_of_residual_output_probabilities[:first_index_to_remove_form_matrix],
                self.sum_of_residual_output_probabilities[first_index_to_not_remove_anymore_form_matrix:]
            )
        )
        # remove the column
        self.sum_of_residual_output_probabilities = torch.cat(
            (
                self.sum_of_residual_output_probabilities[:, :first_index_to_remove_form_matrix],
                self.sum_of_residual_output_probabilities[:, first_index_to_not_remove_anymore_form_matrix:]
            ),
            dim=1
        )
        # and to instances/mean seen:
        self.nr_of_instances_seen_for_cov = torch.cat(
            (
                self.nr_of_instances_seen_for_cov[:idx_to_remove_relative_to_matrix],
                self.nr_of_instances_seen_for_cov[idx_to_remove_relative_to_matrix + 1:]
            )
        )
        self.mean_of_output_probabilities = torch.cat(
            (
                self.mean_of_output_probabilities[:first_index_to_remove_form_matrix],
                self.mean_of_output_probabilities[first_index_to_not_remove_anymore_form_matrix:]
            )
        )

    def _delete_layer(self, layer_index: int) -> bool:
        self._remove_layer_from_covariance_matrix(layer_index)
        self.model._prune_layer_by_vote_removal(layer_index)
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())
        return True

    def _add_layer(self) -> bool:
        self._add_layer_to_covariance_matrix()
        self.model._add_layer()
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())
        return True

    def _add_node(self, layer_index: int) -> bool:
        self.model._add_node(layer_index)
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())
        return True

    def _delete_node(self, layer_index: int, node_index: int) -> bool:
        self.model._delete_node(layer_index, node_index)
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())
        return True

    def _reset_learning_rate(self):
        if not self.learning_rate_progression:
            # if the flag is not set the learning rate does not change
            return 
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    @property
    def learning_rate(self):
        if self.learning_rate_progression:
            return self.__learning_rate()
        else:
            return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value: Union[float | BaseLearningRateProgression]) -> None:
        if isinstance(value, float):
            self.__learning_rate = value
            self.learning_rate_progression = False

        elif isinstance(value, BaseLearningRateProgression):
            self.__learning_rate = value
            self.__learning_rate.classifier = self
            self.learning_rate_progression = True

        else:
            raise TypeError(f"the given learning rate {value} is of unsupported type {type(value)}")

    @property
    def nr_of_instances_seen(self) -> int:
        return self._nr_of_instances_seen
    
    @nr_of_instances_seen.setter
    def nr_of_instances_seen(self, value: int) -> None:
        self._nr_of_instances_seen = value

    @property
    def state_dict(self) -> Dict[str, Any]:
        # todo: learning rate progression
        eval_file = BytesIO()
        JPickler(eval_file).dump(self.adl_evaluator.__getstate__())
        eval_file.seek(0)
        # eval_bytes = eval_file.read()

        drift_detector_file = BytesIO()
        JPickler(drift_detector_file).dump(self.drift_detector.__getstate__())
        drift_detector_file.seek(0)
        # detector_bytes = eval_file.read()

        return {
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_fn': self.loss_function,
            'lr': self.learning_rate,
            # 'lr_progression': self.learning_rate_progression,
            'device': self.device,
            'drift_warning_data': self.drift_warning_data,
            'drift_warning_label': self.drift_warning_label,
            'mci_threshold_for_pruning': self.mci_threshold_for_layer_pruning,
            'mean_of_bias_squared': self.mean_of_bias_squared,
            'sum_of_bias_squared_residuals_squared': self.sum_of_bias_squared_residuals_squared,
            'standard_deviation_of_bias_squared': self.standard_deviation_of_bias_squared,
            'mean_of_variance_squared_squared': self.mean_of_variance_squared_squared,
            'sum_of_variance_squared_residuals_squared': self.sum_of_variance_squared_residuals_squared,
            'standard_deviation_of_variance_squared_squared': self.standard_deviation_of_variance_squared_squared,
            'minimum_of_mean_of_bias_squared': self.minimum_of_mean_of_bias_squared,
            'minimum_of_standard_deviation_of_bias_squared': self.minimum_of_standard_deviation_of_bias_squared,
            'minimum_of_mean_of_variance_squared_squared': self.minimum_of_mean_of_variance_squared_squared,
            'minimum_of_standard_deviation_of_variance_squared_squared': self.minimum_of_standard_deviation_of_variance_squared_squared,
            'random_seed': self.random_seed,
            'nr_of_instances_tracked_in_aggregates_of_bias_and_variance': self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance,

            'adl_evaluator': eval_file,
            'drift_detector': drift_detector_file,
            'drift_criterion': self.drift_criterion_switch,

            'sum_of_residual_output_probabilities': self.sum_of_residual_output_probabilities,
            'mean_of_output_probabilities': self.mean_of_output_probabilities,
            'nr_of_instances_seen_for_cov': self.nr_of_instances_seen_for_cov,
            'nr_of_instances_seen': self.nr_of_instances_seen,
        }

    @state_dict.setter
    def state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.loss_function = state_dict['loss_fn']
        self.learning_rate = state_dict['lr']
        # self.learning_rate_progression = state_dict['lr_progression']
        self.device = state_dict['device']
        self.drift_warning_data = state_dict['drift_warning_data']
        self.drift_warning_label = state_dict['drift_warning_label']
        self.mci_threshold_for_layer_pruning = state_dict['mci_threshold_for_pruning']
        self.mean_of_bias_squared = state_dict['mean_of_bias_squared']
        self.sum_of_bias_squared_residuals_squared = state_dict['sum_of_bias_squared_residuals_squared']
        self.standard_deviation_of_bias_squared = state_dict['standard_deviation_of_bias_squared']
        self.mean_of_variance_squared_squared = state_dict['mean_of_variance_squared_squared']
        self.sum_of_variance_squared_residuals_squared = state_dict['sum_of_variance_squared_residuals_squared']
        self.standard_deviation_of_variance_squared_squared = state_dict['standard_deviation_of_variance_squared_squared']
        self.minimum_of_mean_of_bias_squared = state_dict['minimum_of_mean_of_bias_squared']
        self.minimum_of_standard_deviation_of_bias_squared = state_dict['minimum_of_standard_deviation_of_bias_squared']
        self.minimum_of_mean_of_variance_squared_squared = state_dict['minimum_of_mean_of_variance_squared_squared']
        self.minimum_of_standard_deviation_of_variance_squared_squared = state_dict['minimum_of_standard_deviation_of_variance_squared_squared']
        self.random_seed = state_dict['random_seed']
        self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance = state_dict['nr_of_instances_tracked_in_aggregates_of_bias_and_variance']

        self.adl_evaluator.__dict__.update(JUnpickler(state_dict['adl_evaluator']).load())

        self.drift_detector.__dict__.update(JUnpickler(state_dict['drift_detector']).load())

        self.drift_criterion_switch = state_dict['drift_criterion']

        self.sum_of_residual_output_probabilities = state_dict['sum_of_residual_output_probabilities']
        self.mean_of_output_probabilities = state_dict['mean_of_output_probabilities']
        self.nr_of_instances_seen_for_cov = state_dict['nr_of_instances_seen_for_cov']
        self.nr_of_instances_seen = state_dict['nr_of_instances_seen']
