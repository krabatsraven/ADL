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

        def step(self, true_label):
            # optimizer step of the super.optimizer that optimizes the parameters of the network
            super().step()

            # adjust voting weights
            # self._adjust_weights(true_label, step_size=0.01)

            # todo: high level learning
            # todo: low level learning

        def _adjust_weights(self, true_label: torch.Tensor, step_size: float):

            # find the indices of the results that where correctly predicted
            correctly_predicted_layers_indices = np.where(torch.argmax(network.layer_results, dim = 1) == true_label)
            correctly_predicted_layers_mask = np.zeros(self.network.layer_result_keys.shape, dtype=bool)
            correctly_predicted_layers_mask[correctly_predicted_layers_indices] = True

            # if layer predicted correctly increase weight correction factor p^(l) by step_size "zeta"
            # p^(l) = p^(l) + step_size
            keys_of_correctly_predicted_layers = self.network.layer_result_keys[correctly_predicted_layers_mask]
            increased_correction_weights = {key: self.network.weight_correction_factor[key] + step_size for key in keys_of_correctly_predicted_layers}
            self.network.weight_correction_factor.update(increased_correction_weights)
            # if layer predicted erroneous decrease weight correction factor p^(l) by step size
            # p^(l) = p^(l) - step_size
            keys_of_incorrectly_predicted_layers = self.network.layer_result_keys[~correctly_predicted_layers_mask]
            decreased_correction_weights = {key: self.network.weight_correction_factor[key] - step_size for key in keys_of_incorrectly_predicted_layers}
            self.network.weight_correction_factor.update(decreased_correction_weights)

            # adjust weight of layer l:

            # if layer l was correct increase beta:
            # increase beta^(l) while assuring that 0 <= beta^(l) <= 1 by
            # beta^(l) = min((1 + p^(l)) * beta^(l), 1)
            increased_voting_weights = {
                key: 
                    min(
                        (1 + self.network.weight_correction_factor[key]) * self.network.voting_weights[key],
                        1
                    ) 
                for key in keys_of_correctly_predicted_layers
            }
            self.network.voting_weights.update(increased_voting_weights)

            # if layer l was correct decrease beta:
            # beta^(l) = p^(l) * beta^(l)
            decreased_voting_weights = {
                key: self.network.weight_correction_factor[key] * self.network.voting_weights[key]
                for key in keys_of_incorrectly_predicted_layers
            }
            self.network.voting_weights.update(decreased_voting_weights)

        def _high_lvl_learning(self):
            raise NotImplementedError

        def _low_lvl_learning(self):
            raise NotImplementedError

    return ADLOptimizer()