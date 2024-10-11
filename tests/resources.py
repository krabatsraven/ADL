from itertools import combinations
import random

import torch

from AutoDeepLearner import AutoDeepLearner

optimizer_choices = [torch.optim.SGD, torch.optim.Adam]
learning_rate_combinations = list(combinations([0.0001, 0.01, 0.1, 1], 2))
trainings_steps = range(2, 5)

def random_initialize_model(model: AutoDeepLearner, iteration_count: int) -> AutoDeepLearner:
    for _ in range(iteration_count):
        dice = random.choice(range(1, 5))

        match dice:
            case 1:
                model._add_layer()

                # the add_layer function initialises the voting weight with 0
                # not all voting weight can be zero: solution here: randomly assign a value between zero and one
                last_added_layer_idx = len(model.layers) - 1
                model._AutoDeepLearner__set_voting_weight(last_added_layer_idx, random.uniform(0, 1))

                # and normalise the voting weights
                model._normalise_voting_weights()

            case 2:
                layer_choice = int(random.choice(model.get_voting_weight_keys()))
                model._add_node(layer_choice)
            case 3:
                if len(model.get_voting_weight_keys()) > 2:
                    layer_choice = int(random.choice(model.get_voting_weight_keys()))
                    model._prune_layer_by_vote_removal(layer_choice)
                else:
                    continue
            case 4:
                if len(model.get_voting_weight_keys()) > 2:
                    layer_choice = int(random.choice(model.get_voting_weight_keys()))
                    if model.layers[layer_choice].weight.size()[0] > 2:
                        node_choice = random.choice(range(model.layers[layer_choice].weight.size()[0]))
                        model._delete_node(layer_choice, node_choice)
                    else:
                        continue
                else:
                    continue

    return model