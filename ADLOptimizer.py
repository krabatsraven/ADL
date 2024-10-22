import numpy as np
import torch.optim

from AutoDeepLearner import AutoDeepLearner


def create_adl_optimizer(
        network: AutoDeepLearner,
        optimizer: type(torch.optim.Optimizer),
        learning_rate: float,
        **kwargs):
    """
    creates the optimizer object that extents the pytorch optimizer of the provided type and returns it
    this optimizer only adds to the step function, where in actual optimization of the parameters is done by the provided optimizer
    :param network: the network that is supposed to be optimized
    :param optimizer: the type of pytorch optimizer that is to be used (tested for torch.optim.SGD and torch.optim.ADAM)
    :param learning_rate: the learning rate of the optimizer, note that not all possible optimizers have the kwarg lr, so that might error
    :param kwargs: arguments to initialize the optimizer object
    :return: the optimizer object of the provided type
    """

    class ADLOptimizer(optimizer):

        def __repr__(self):
            return f"ADLOptimizer(network={network}, optimizer={optimizer}, learning_rate={learning_rate})"

        def __init__(self):
            super().__init__(network.parameters(), lr=learning_rate, **kwargs)
            self.network = network
            self.learning_rate = learning_rate

            self.stop_flag = False

        def step(self, true_label):
            # optimizer step of the super.optimizer that optimizes the parameters of the network
            super().step()

            # adjust voting weights
            self._adjust_weights(true_label, step_size=self.learning_rate)

            # todo: high level learning
            # todo: low level learning

        def _adjust_weights(self, true_label: torch.Tensor, step_size: float):
            # find the indices of the results that where correctly predicted
            correctly_predicted_layers_indices = np.where(torch.argmax(network.layer_results, dim = 1) == true_label)
            correctly_predicted_layers_mask = np.zeros(self.network.layer_result_keys.shape, dtype=bool)
            correctly_predicted_layers_mask[correctly_predicted_layers_indices] = True

            weight_correction_factor_values = self.network.get_weight_correction_factor_values()
            step_size_tensor = torch.tensor(step_size, dtype=torch.float)

            # if layer predicted correctly increase weight correction factor p^(l) by step_size "zeta"
            # p^(l) = p^(l) + step_size
            weight_correction_factor_values[correctly_predicted_layers_mask] += step_size_tensor
            # if layer predicted erroneous decrease weight correction factor p^(l) by step size
            # p^(l) = p^(l) - step_size
            weight_correction_factor_values[~correctly_predicted_layers_mask] -= step_size_tensor

            # assure that epsilon <= p^(l) <= 1
            weight_correction_factor_values = torch.where(
                weight_correction_factor_values > self.network.upper_weigth_correction_factor_boarder,
                self.network.upper_weigth_correction_factor_boarder,
                weight_correction_factor_values
            )
            weight_correction_factor_values = torch.where(
                weight_correction_factor_values < self.network.lower_weigth_correction_factor_boarder,
                self.network.lower_weigth_correction_factor_boarder,
                weight_correction_factor_values
            )

            # create mapping of values and update parameter dict of network
            weight_correction_factor_items = {
                key: value.item()
                for key, value in zip(
                    self.network.weight_correction_factor.keys(),
                    weight_correction_factor_values
                )
            }
            self.network.weight_correction_factor.update(weight_correction_factor_items)

            # adjust voting weight of layer l:
            voting_weight_values = self.network.get_voting_weight_values()

            # if layer l was correct increase beta:
            # beta^(l) = (1 + p^(l)) * beta^(l)
            voting_weight_values[correctly_predicted_layers_mask] *= (1 + weight_correction_factor_values[correctly_predicted_layers_mask])

            # if layer l was incorrect decrease beta:
            # beta^(l) = p^(l) * beta^(l)
            voting_weight_values[~correctly_predicted_layers_mask] *= weight_correction_factor_values[~correctly_predicted_layers_mask]

            # while assuring that 0 <= beta^(l) <= 1
            voting_weight_values = torch.where(
                voting_weight_values > self.network.upper_voting_weight_boarder,
                self.network.upper_voting_weight_boarder,
                voting_weight_values
            )
            voting_weight_values = torch.where(
                voting_weight_values < self.network.lower_voting_weigth_boarder,
                self.network.lower_voting_weigth_boarder,
                voting_weight_values
            )

            # create mapping of values and update parameter dict of network
            voting_weight_items = {
                key: value.item()
                for key, value in zip(
                    self.network.voting_weights.keys(),
                    voting_weight_values
                )
            }
            self.network.voting_weights.update(voting_weight_items)

        def _high_lvl_learning(self):
            raise NotImplementedError

        def _low_lvl_learning(self):
            raise NotImplementedError

    return ADLOptimizer()