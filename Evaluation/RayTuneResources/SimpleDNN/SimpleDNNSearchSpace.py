from itertools import product

import numpy as np
from ray import tune


def SimpleDNNSearchSpace(stream_name: str, nr_of_hidden_layers: int = 5, nr_of_neurons: int = 2**12):
    """
    creates a search space for the SimpleDNN model
    that has no more than nr_of_hidden_layers many linear layers
    and in total not more than 2*nr_of_neurons many nodes
    we assume that nr_of_neurons is greater than 256, and nr_of_hidden_layers positive.
    """
    return {
        "lr": tune.loguniform(1e-4, 5e-1),
        "model_structure": tune.choice(
            [
                list(perm)
                for h in range(1, nr_of_hidden_layers + 1)
                for perm in product(list(map(int, 2 ** np.arange(int(np.ceil(np.log2(nr_of_neurons))) + 1))), repeat=h)
                if np.sum(perm) <= 2**np.ceil(np.log2(nr_of_neurons))
            ]
        ),
        'stream': tune.grid_search([stream_name])
    }
