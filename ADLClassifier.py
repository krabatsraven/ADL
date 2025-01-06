import time
from typing import Optional

import numpy as np
import torch
from capymoa.base import Classifier
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import ClassificationEvaluator
from capymoa.stream import Schema
from codecarbon import track_emissions
from torch import nn
from torch.optim import Optimizer

from AutoDeepLearner import AutoDeepLearner


class ADLClassifier(Classifier):
    def __init__(
            self,
            schema: Optional[Schema] = None,
            random_seed: int = 1,
            nn_model: Optional[AutoDeepLearner] = None,
            optimizer: Optional[Optimizer] = None,
            loss_fn=nn.CrossEntropyLoss(),
            device: str = ("cpu"),
            lr: float = 1e-3,
            drift_detector: BaseDriftDetector = ADWIN(delta=0.001),
            drift_criterion: str = "accuracy",
            mci_threshold_for_layer_pruning: float = 10**-7,
            nr_of_results_kept_for_covariance_calculation: int = 10
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


        self.evaluator = ClassificationEvaluator(self.schema, window_size=1)

        # keep track of the shape of the model for each seen instance for later evaluation
        self.record_of_model_shape = {
            "nr_of_layers": [],
            "shape_of_hidden_layers": [],
            "active_layers": []
        }

        self.drift_detector: BaseDriftDetector = drift_detector
        self.__drift_criterion_switch = drift_criterion

        # data and label which triggered a drift warning in the past
        self.drift_warning_data: Optional[torch.Tensor] = None
        self.drift_warning_label: Optional[torch.Tensor] = None

        # user chosen threshold for the mci value that guards the layer pruning
        self.mci_threshold_for_layer_pruning = mci_threshold_for_layer_pruning

        # result data that are kept to calculate the covariance matrix of output nodes of different output layers
        # todo: eliminate by recursive calculation of covariant matrix
        self.results_of_all_hidden_layers_kept_for_cov_calc: Optional[torch.Tensor] = None
        self.nr_of_results_kept_for_covariance_calculation = nr_of_results_kept_for_covariance_calculation

        # values that are tracking the bias and variance of the model to inform node growing/pruning decisions
        self.nr_of_instances_tracked_in_aggregates_of_bias_and_variance: torch.Tensor = torch.tensor(0, dtype=torch.int, requires_grad=False)

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

        # todo: create clean version without evaluation: remove "total time in loop"
        self.total_time_in_loop = 0


    def __str__(self):
        return "ADLClassifier"

    def CLI_help(self):
        return str('schema=None, random_seed=1, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device=("cpu"), lr=1e-3 evaluator=ClassificationEvaluator(), drift_detector=ADWIN(delta=0.001), drift_criterion="accuracy"')

    def set_model(self, instance):
        if self.schema is not None:
            self.model: AutoDeepLearner = AutoDeepLearner(
                nr_of_features = self.schema.get_num_attributes(),
                nr_of_classes = self.schema.get_num_classes()
            ).to(self.device)

        elif instance is not None:
            moa_instance = instance.java_instance.getData()

            self.model: AutoDeepLearner = AutoDeepLearner(
                nr_of_features = moa_instance.get_num_attributes(),
                nr_of_classes = moa_instance.get_num_classes()
            ).to(self.device)

    def train(self, instance):
        if self.model is None:
            self.set_model(instance)

        # todo: create clean version without evaluation: remove "record of model shape"
        self.__update_record_of_model_shape()
        self.__train(instance)

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
    # todo: create clean version without evaluation: remove "track emissions"
    @track_emissions(offline=True, country_iso_code="DEU")
    def __train(self, instance):
        X = torch.tensor(instance.x, dtype=torch.float32)
        y = torch.tensor(instance.y_index, dtype=torch.long)
        # set the device and add a dimension to the tensor
        X, y = torch.unsqueeze(X.to(self.device), 0), torch.unsqueeze(y.to(self.device),0)

        # add data to tracked_statistical variables:
        self.model.add_data_to_statistical_variables(X)

        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_function(pred, y)

        # Backpropagation
        # todo: q: winning layer training? when?
        loss.backward()
        self.optimizer.step()

        self.evaluator.update(y.item(), torch.argmax(pred).item())
        # self.evaluator.metrics().append(len(self.model.layers))
        # self.evaluator.result_windows[-1].append(len(self.model.layers))
        self.drift_detector.add_element(self.__drift_criterion(y, pred))

        # todo: these next three lines are part of the evaluation and should not be part of co2 emissions

        self._adjust_weights(true_label=y, step_size=self.learning_rate)
        # todo: # 71
        # todo: # 72
        self._high_lvl_learning(true_label=y, data=X)
        self._low_lvl_learning(true_label=y)
        self.optimizer.zero_grad()

    # todo: new name for _adjust_weight()
    def _adjust_weights(self, true_label: torch.Tensor, step_size: float):
        # find the indices of the results that where correctly predicted
        correctly_predicted_layers_indices = torch.where(torch.argmax(self.model.layer_results, dim = 1) == true_label)
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

    def _high_lvl_learning(self, true_label: torch.Tensor, data: torch.Tensor):
        self.__high_level_learning_pruning_hidden_layers()
        self.__high_level_learning_grow_new_hidden_layers(data, true_label)

    def __high_level_learning_pruning_hidden_layers(self):
        # prune highly correlated layers:
        # ------------------------------

        # find correlated layers:
        # compare the predictions of the layer pairwise
        # layer_results are the stacked result_i, where result_i stems from output layer i
        # meaning the shape is (nr_of_layers, nr_of_classes)
        n = self.model.active_layer_keys().size(dim=0)
        m = self.model.output_size
        # to get all covariance values at once we reshape to (nr_of_layers * nr_of_classes, x),
        # where x is the nr of instances seen
        # meaning: (layer_1_class_1_prob, layer_1_class_2_prob, ..., layer_1_class_m_prob, layer_2_class_1_prob, ...), (2. instance)
        current_results = self.model.layer_results.flatten().reshape(-1, n * m).transpose(0,1).detach()
        current_results.requires_grad_(False)

        # todo: # 82
        if self.results_of_all_hidden_layers_kept_for_cov_calc is not None:
            # concat all results of all hidden layers
            self.results_of_all_hidden_layers_kept_for_cov_calc = torch.cat(
                tensors=(self.results_of_all_hidden_layers_kept_for_cov_calc.clone(), current_results),
                dim=1
            )
            # only keep the self.nr_of_results_kept_for_covariance_calculation last results
            self.results_of_all_hidden_layers_kept_for_cov_calc = self.results_of_all_hidden_layers_kept_for_cov_calc[
                                                                  :,
                                                                  self.results_of_all_hidden_layers_kept_for_cov_calc.size(dim=1)
                                                                  - self.nr_of_results_kept_for_covariance_calculation:
                                                                  ]
        else:
            self.results_of_all_hidden_layers_kept_for_cov_calc = current_results

        if self.model.active_layer_keys().size(dim=0) < 2 or self.results_of_all_hidden_layers_kept_for_cov_calc.size(dim=-1) == 1:
            # if there are less than 2 active layers we cannot find a correlated layer
            # also if there are less than 2 different results for all layers (right after changing of a layer) 
            # we would divide by 0 (corrected statistical covariance)
            return

        all_covariances_matrix = torch.cov(self.results_of_all_hidden_layers_kept_for_cov_calc)
        variances = torch.diag(all_covariances_matrix)
        mci_values = -1 * torch.ones((n, n, m))

        this_loop_start = time.time_ns()
        # Calculate MCI for each pair of layers
        # todo: # 81
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

        this_loop_stop = time.time_ns()
        time_in_this_loop = this_loop_stop - this_loop_start
        self.total_time_in_loop += time_in_this_loop

        # find the correlated pairs by comparing the max mci for each pairing against a user chosen threshold
        mci_max_values = torch.max(mci_values, dim=2).values
        # the pair (0, 1) and (1, 0) should only appear once
        # later we prioritize to delete layer j before layer i if i < j
        # the maximal mci for a layer pair has to be smaller than the users threshold
        correlated_pairs = ((torch.logical_and(0 <= mci_max_values, mci_max_values < self.mci_threshold_for_layer_pruning))
                            .nonzero()
                            .sort(dim = -1)[0]
                            .unique(dim=0, sorted=True, return_counts=False, return_inverse=False)
                            .flip(dims=(0,))
                            )

        # prune them
        active_layers_at_the_start = self.model.active_layer_keys().clone()
        for index_tensor_of_layer_i, index_tensor_of_layer_j in correlated_pairs:
            # transform the relative indices of correlated_pairs to the absolute indices of the active layers:
            index_of_layer_i = active_layers_at_the_start[index_tensor_of_layer_i].item()
            index_of_layer_j = active_layers_at_the_start[index_tensor_of_layer_j].item()
            # todo: issue #80
            if self.model.output_layer_with_index_exists(index_of_layer_i) and self.model.output_layer_with_index_exists(index_of_layer_j):
                # if both layers still exist:
                if self.model.get_voting_weight(index_of_layer_i) < self.model.get_voting_weight(index_of_layer_j):
                    # layer_i has the smaller voting weight and gets pruned
                    self.model._prune_layer_by_vote_removal(index_of_layer_i)
                else:
                    # voting_weight(j) <= voting_weight(i) => prune j
                    # as i <= j is guaranteed by beforehand sorting
                    # we prune j if voting_weight(i) == voting_weight(j)
                    self.model._prune_layer_by_vote_removal(index_of_layer_j)

                self.results_of_all_hidden_layers_kept_for_cov_calc = None

        # update the optimizer after pruning:
        self.optimizer.param_groups[0]['params'] = list(self.model.parameters())

    def __high_level_learning_grow_new_hidden_layers(self, data: torch.Tensor, true_label: torch.Tensor):
        # grow the hidden layers to accommodate concept drift when it happens:
        # -------------------------------------------------------------------
        if self.drift_detector.detected_change():
            # stack saved data if there is any onto the current instance to train with both
            if self.drift_warning_data is not None:
                data = torch.stack((self.drift_warning_data, data))
                true_label = torch.stack((self.drift_warning_label, true_label))

            # add layer
            active_layers_before_adding = self.model.active_layer_keys().tolist()
            self.model._add_layer()
            self.results_of_all_hidden_layers_kept_for_cov_calc = None
            # update optimizer after adding a layer
            self.optimizer.param_groups[0]['params'] = list(self.model.parameters())

            # train the layer:
            # todo: question #69
            # freeze the parameters not new:
            # originally they create a new network with only one layer and train the weights there
            # can we just delete the gradients of all weights not in the new layer?
            # train by gradient
            # disable all active layers that were not just added:
            self.model._disable_layers_for_training(active_layers_before_adding)
            pred = self.model.forward(data)
            loss = self.loss_function(pred, true_label)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
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
        self.__update_mu_and_s_of_bias_and_var(bias_of_winning_layer_squared, variance_of_winning_layer_squared)

        # \kappa
        kappa = 1.3 * torch.exp(-bias_of_winning_layer_squared) + 0.7
        # if criterion 1: µ_bias + s_bias >= min_µ_bias + \kappa * min_s_bias
        if (self.mean_of_bias_squared + self.standard_deviation_of_bias_squared 
                >= self.minimum_of_mean_of_bias_squared + kappa * self.minimum_of_standard_deviation_of_bias_squared):
            # add node to winning layer
            self.model._add_node(winning_layer)
            # return: we don't add and shrink in the same turn
            return

        # \chi
        xi = 1.3 * torch.exp(-variance_of_winning_layer_squared) + 0.7
        # if criterion 2:  µ_var + s_var >= min_µ_var + 2 * \chi * min_s_var
        if (self.mean_of_variance_squared_squared + self.standard_deviation_of_variance_squared_squared 
                >= self.minimum_of_mean_of_variance_squared_squared + 2 * xi *self.minimum_of_standard_deviation_of_variance_squared_squared):
            # remove the least contributing node from the winning layer
            self.model._delete_node(winning_layer, idx_of_least_contributing_winning_layer_node)

            # if the node is deleted the minima are reset:
            self.minimum_of_mean_of_bias_squared = self.mean_of_bias_squared
            self.minimum_of_standard_deviation_of_bias_squared = self.standard_deviation_of_bias_squared
            self.minimum_of_mean_of_variance_squared_squared = self.mean_of_variance_squared_squared
            self.minimum_of_standard_deviation_of_variance_squared = self.standard_deviation_of_variance_squared_squared

    def __update_mu_and_s_of_bias_and_var(self, bias_squared: torch.Tensor, variance_squared: torch.Tensor) -> None:
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

    def __drift_criterion(self, true_label: torch.Tensor, prediction: torch.Tensor) -> float:
        match self.__drift_criterion_switch:
            case "accuracy":
                # use accuracy to univariant detect concept drift
                return self.evaluator.accuracy()

            case "loss":
                # use the loss to univariant detect concept drift
                return self.loss_function(true_label, prediction)

    def __update_record_of_model_shape(self):
        self.record_of_model_shape["nr_of_layers"].append(len(self.model.layers))
        self.record_of_model_shape["shape_of_hidden_layers"].append([tuple(list(h.weight.size()).__reversed__()) for h in self.model.layers])
        self.record_of_model_shape["active_layers"].append(self.model.active_layer_keys().numpy())
