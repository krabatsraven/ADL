from Evaluation.EvaluationFunctions import _test_example

# 82% acc: {'learner': ('vectorized', 'winning_layer', 'decoupled_lrs'), 'stream': 'electricity', 'lr': 0.17037433308206834, 'layer_weight_learning_rate': 0.0051048969488651065, 'adwin-delta': 2.2019797256079463e-05, 'mci': 2.3105218391180886e-07, 'grace_period': (32, 'global_grace'), 'loss_fn': 'NLLLoss'}

if __name__ == "__main__":
    _test_example(name="new_test_function")

    # todo: effects compare normalization, one hot_encoding