from typing import Tuple

from ray import tune

def ADLSearchSpace(stream_name: str, learner: Tuple[str, ...] = ('vectorized', 'winning_layer', 'decoupled_lrs')):
    return {
        'learner': tune.choice(
            [
                learner
            ]
        ),
        'stream': tune.choice(
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
                for grace_period in [1, 4, 8, 16, 32, 64, 128, 256, 512]
            ]
        ),
        'loss_fn': tune.choice(
            [
                # 'CrossEntropyLoss',
                'NLLLoss'
            ]
        )
    }