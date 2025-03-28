from ray import tune

from Evaluation._config import STREAM_STRINGS

UnStableStableSearchSpace = {
    'learner': tune.choice([('input_preprocessing', 'vectorized', 'winning_layer', 'decoupled_lrs')]),
    'stream': tune.choice(STREAM_STRINGS),
    'lr': tune.loguniform(1e-4, 5e-1),
    'layer_weight_learning_rate': tune.loguniform(1e-4, 5e-1),
    'adwin-delta': tune.loguniform(1e-7, 1e-3),
    'mci': tune.loguniform(1e-7, 1e-5),
    'grace_type': tune.choice(["global_grace", "layer_grace"]),
    'grace_period': tune.qrandint(1,500),
    'loss_fn': tune.choice(['NLLLoss'])
}