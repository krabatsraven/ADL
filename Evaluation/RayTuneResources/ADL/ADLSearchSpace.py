import torch
from ray import tune
from torch import nn


ADLSearchSpace = {
    'learner': tune.grid_search(
        [
            ('vectorized', 'winning_layer'),
            # ('winning_layer',),
            # ('vectorized',)
        ]
    ),
    'stream': tune.grid_search(
        [
            # todo: replace with the others
            '/home/david/PycharmProjects/ADL/data/electricity_tiny.arff'
        ]
    ),
    # todo: add progressions
    'lr': tune.loguniform(1e-4, 5e-1),
    'adwin-delta': tune.loguniform(1e-7, 1e-3),
    'mci': tune.loguniform(1e-7, 1e-5),
    'grace_period': tune.choice(
        [
            (grace_period, is_global) if grace_period is not None else None
            for is_global in ["globel_grace", "layer_grace"]
            for grace_period in [None, 4, 8, 16, 32]
        ]
    ),
    'loss_fn': tune.grid_search(
        [
            nn.CrossEntropyLoss(),
            lambda predicted_props, truth: nn.NLLLoss()(torch.log(predicted_props), truth)
        ]
    )
}