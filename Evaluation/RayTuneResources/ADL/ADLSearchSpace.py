import torch
from ray import tune
from torch import nn

from ADLClassifier.Resources.NLLLoss import NLLLoss

def ADLSearchSpace(stream_name: str):
    return {
        'learner': tune.grid_search(
            [
                ('vectorized', 'winning_layer'),
                ('vectorized', 'winning_layer', 'decoupled_lrs'),
                # ('winning_layer',),
                # ('vectorized',)
            ]
        ),
        'stream': tune.grid_search(
            [
                stream_name
            ]
        ),
        # todo: add progressions
        'lr': tune.loguniform(1e-4, 5e-1),
        'layer_weight_learning_rate': tune.loguniform(1e-4, 5e-1),
        'adwin-delta': tune.loguniform(1e-7, 1e-3),
        'mci': tune.loguniform(1e-7, 1e-5),
        'grace_period': tune.choice(
            [
                (grace_period, is_global) if grace_period is not None else None
                for is_global in ["global_grace", "layer_grace"]
                for grace_period in [None, 4, 8, 16, 32]
            ]
        ),
        'loss_fn': tune.grid_search(
            [
                'CrossEntropyLoss',
                'NLLLoss'
            ]
        )
    }