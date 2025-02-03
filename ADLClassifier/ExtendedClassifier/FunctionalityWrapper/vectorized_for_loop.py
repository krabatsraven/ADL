import time
from typing import List, Tuple

import torch

from ADLClassifier.BaseClassifier import ADLClassifier


def vectorized_for_loop(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):
    """
    Vectorizes the triple for loop in the calculation of the mci score to determine correlated pairs of output layers
     in the given/decorated class of ADLClassifier
    :param adl_classifier: the class of ADLClassifier, whose mci score calculation should be vectorized
    :return: a class of ADLClassifier, whose mci score calculation is vectorized
    """
    class WithoutForLoopWrapper(adl_classifier):
        """
        A class of ADLClassifier with a vectorized mci score calculation
        """

        def __str__(self):
            return f"{super().__str__()}WithoutForLoop"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithoutForLoop"

        def _get_correlated_pairs_of_output_layers(self) -> List[Tuple[int, int]]:
            # find correlated layers:
            # compare the predictions of the layer pairwise
            # layer_results are the stacked result_i, where result_i stems from output layer i
            # meaning the shape is (nr_of_layers, nr_of_classes)

            for row in self.model.layer_results:
                self._update_covariance_with_one_row(row)

            comb = torch.combinations(torch.arange(self.model.nr_of_active_layers))
            i, j = comb[:, 0], comb[:, 1]
            cov_mat = self._covariance_of_output_nodes()
            var_i, var_j = cov_mat[i, i], cov_mat[j, j]
            cov_ij = cov_mat[i, j]
            pearson_corr = cov_ij / torch.sqrt(var_i * var_j)
            mci_2 = torch.absolute(0.5 * (var_i + var_j) - torch.sqrt((var_i + var_j)**2 - 4 * var_i * var_j * (1 - pearson_corr**2)))
            mci_2_max_values = mci_2.max(dim=1).values
            # find the correlated pairs by comparing the max mci for each pairing against a user chosen threshold
            correlated_pairs = comb[(mci_2_max_values < self.mci_threshold_for_layer_pruning).nonzero()]

            out = [(self.model.active_layer_keys()[pair.squeeze()[0]].item(), self.model.active_layer_keys()[pair.squeeze()[1]].item()) for pair in correlated_pairs]
            return out

        def __init_cov_variables__(self) -> None:
            # to later calculate the mci for layer removal
            # ---------------------------------------------------
            # the sum of residuals:
            # C_n   := sum_{i=1}^n(x_i - \bar{x_n})(y_i - \bar{y_n})
            # of the class probabilities of active layers:
            self.sum_of_output_probability_deviation_products: torch.Tensor = torch.zeros(
                (self.model.nr_of_active_layers, self.model.nr_of_active_layers, self.model.output_size)
            )
            # the mean class probabilities for each active layer
            self.mean_of_output_probabilities: torch.Tensor = torch.zeros(
                (self.model.nr_of_active_layers, self.model.output_size)
            )
            # nr of instances seen in each layer
            self.nr_of_instances_seen_for_cov = torch.zeros(self.model.nr_of_active_layers, dtype=torch.int)

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
            dx = layer_result - self.mean_of_output_probabilities
            # dx = x - mean_x (of n-1 instances)

            # \bar x_n = \bar {x_{n-1}} + \frac{x_n - \bar{x_{n-1}}}{n} = mean_x + dx / n
            self.mean_of_output_probabilities = (
                    self.mean_of_output_probabilities
                    +
                    (
                            dx
                            /
                            self.nr_of_instances_seen_for_cov.unsqueeze(1)
                    )
            )
    
            # meany += (y - meany) / n
            # C += dx * (y - meany)
            # C_n   := sum_{i=1}^n(x_i - \bar{x_n})(y_i - \bar{y_n})
            #       = C_{n-1} + (x_n - \bar{x_n})(y_n - \bar{y_{n - 1}})
            #       = C_{n-1} + (x_{n} - \bar{x_{n - 1}})(y_n - \bar{y_{n}})
            self.sum_of_output_probability_deviation_products = (
                    self.sum_of_output_probability_deviation_products
                    +
                    (
                        dx
                        .unsqueeze(1)
                        .mul(
                            (
                                (layer_result - self.mean_of_output_probabilities)
                                .unsqueeze(0)
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
            un_squeezed_bechel_corrected_count: torch.Tensor = (self.nr_of_instances_seen_for_cov - 1).unsqueeze(0).unsqueeze(2)
            return (
                (self.sum_of_output_probability_deviation_products / un_squeezed_bechel_corrected_count)
                .where(un_squeezed_bechel_corrected_count > 1, 0)
            )

        def _add_layer_to_covariance_matrix(self):
            # new layers are added: add columns and rows of zeros to the end of cov matrix for each class:
            self.sum_of_output_probability_deviation_products = torch.cat(
                (
                    torch.cat(
                        (
                            self.sum_of_output_probability_deviation_products,
                            torch.zeros((1, self.model.nr_of_active_layers, self.model.output_size))
                        ),
                        dim=0
                    ),
                    torch.zeros((self.model.nr_of_active_layers + 1, 1, self.model.output_size))
                ),
                dim=1
            )
            # and to instances/mean seen:
            self.nr_of_instances_seen_for_cov = torch.cat((self.nr_of_instances_seen_for_cov, torch.zeros(1, dtype=torch.int)))
            self.mean_of_output_probabilities = torch.cat(
                (
                    self.mean_of_output_probabilities,
                    torch.zeros((1, self.model.output_size))
                )
            )

        def _remove_layer_from_covariance_matrix(self, layer_idx: int):
            assert self.model.output_layer_with_index_exists(layer_idx), "we can only remove a layer from the covariance that still has an output layer"
            # if layer x is removed: remove rows i and columns j between x * nr_of_classes <= i, j < (x + 1) * nr_of_classes
            # the index x of the layer to remove is the position of the layer_id in relation to the other active layers
            idx_to_remove_relative_to_matrix = torch.nonzero(self.model.active_layer_keys() == layer_idx).item()
            # remove the row
            self.sum_of_output_probability_deviation_products = torch.cat(
                (
                    self.sum_of_output_probability_deviation_products[:idx_to_remove_relative_to_matrix],
                    self.sum_of_output_probability_deviation_products[idx_to_remove_relative_to_matrix + 1:]
                )
            )
            # remove the column
            self.sum_of_output_probability_deviation_products = torch.cat(
                (
                    self.sum_of_output_probability_deviation_products[:, :idx_to_remove_relative_to_matrix],
                    self.sum_of_output_probability_deviation_products[:, idx_to_remove_relative_to_matrix + 1:]
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
                    self.mean_of_output_probabilities[:idx_to_remove_relative_to_matrix],
                    self.mean_of_output_probabilities[idx_to_remove_relative_to_matrix + 1:]
                )
            )

    WithoutForLoopWrapper.__name__ = f"{adl_classifier.__name__}WithoutForLoop"
    return WithoutForLoopWrapper
