import torch.optim

from AutoDeepLearner import AutoDeepLearner


def create_adl_optimizer(network: AutoDeepLearner, optimizer: type(torch.optim.Optimizer), **kwargs):
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
            return f"ADLOptimizer(network={network}, optimizer={optimizer})"

        def __init__(self):
            super().__init__(network.parameters(), **kwargs)
            self.network = network

        def step(self):
            # optimizer step of the super.optimizer that optimizes the parameters of the network
            super().step()

            # todo: adjust voting weights
            # todo: high level learning
            # todo: low level learning

        def _adjust_weights(self, true_label):
            # todo: add type hint to true_label argument
            # todo: if layer predicted correctly increase weight correction factor p^(l) by step size
            # p^(l) = p^(l) + step_size
            # todo: if layer predicted erroneous decrease weight correction factor p^(l) by step size
            # p^(l) = p^(l) - step_size
            # todo: adjust weight of layer l:
            # todo: if layer l was correct increase beta:
            # increase beta^(l) while assuring that 0 <= beta^(l) <= 1 by
            # beta^(l) = min((1 + p^(l)) * beta^(l), 1)
            # todo: if layer l was correct decrease beta:
            # beta^(l) = p^(l) * beta^(l)
            raise NotImplementedError

        def _high_lvl_learning(self):
            raise NotImplementedError

        def _low_lvl_learning(self):
            raise NotImplementedError

    return ADLOptimizer()