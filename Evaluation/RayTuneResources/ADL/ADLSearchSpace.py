from typing import Tuple

from ray import tune

def ADLSearchSpace(stream_name: str, learner: Tuple[str, ...] = ('vectorized', 'winning_layer', 'decoupled_lrs', 'input_preprocessing')):
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
        'grace_type': tune.choice(["global_grace", "layer_grace"]),
        'grace_period': tune.qrandint(1,500),
        'loss_fn': tune.choice(
            [
                # 'CrossEntropyLoss',
                'NLLLoss'
            ]
        )
    }