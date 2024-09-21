import torch

from ADLOptimizer import adl_optimizer
from AutoDeepLearner import AutoDeepLearner

if __name__ == "__main__":
    adl_network: AutoDeepLearner = AutoDeepLearner(10, 10)

    optimizer: torch.optim.Optimizer = adl_optimizer(adl_network, torch.optim.SGD, lr=0.001, momentum=0.9)
    print(optimizer)
