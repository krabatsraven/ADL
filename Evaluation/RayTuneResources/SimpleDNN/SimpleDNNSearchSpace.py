from itertools import product

from capymoa.datasets import Electricity
from ray import tune


def SimpleDNNSearchSpace(stream_name: str):
    return {  # ②
        "lr": tune.loguniform(1e-4, 5e-1),
        "model_structure": tune.choice(
            [list(perm) for i in range(1, 5) for perm in product([2**8, 2**9, 2**10, 2**11], repeat=i)]
        ),
        'stream': tune.grid_search([stream_name])
    }
