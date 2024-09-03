import random
from typing import List, Tuple

import torch

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


def model_setup() -> Tuple[AutoDeepLearner, List[int]]:
    # set up class with a few layers
    feature_count, class_count = torch.randint(10_000, (2,))
    model = AutoDeepLearner(feature_count, class_count)
    nr_of_layers = 100

    for i in range(nr_of_layers):
        model._add_layer()

    # draw a random amount of indices to add a node too
    layers_to_add_to = [random.randint(0, nr_of_layers) for _ in range(random.randint(0, nr_of_layers))]

    return model, layers_to_add_to


class TestAutoDeepLearnerAddNode:

    def test_add_node_adds_node(self):
        """
        add node should add a node to the right layer
        """

        def add_test_in_layer(model, layer_idx, msg):
            nr_of_out_vectors_before, nr_of_in_vectors_before = model.layers[layer_idx].weight.size()
            model._add_node(layer_idx)

            assert model.layers[layer_idx].weight.size() == (nr_of_out_vectors_before + 1,
                                                             nr_of_in_vectors_before), (msg)

        model, layers_to_add_to = model_setup()

        # testing for random indices:
        for layer_idx in layers_to_add_to:
            add_test_in_layer(model, layer_idx, "Adding a node should have increased the number of out vectors by one, "
                                                "and left the numbers of in vectors the same")

        # testing for random indices:
        for layer_idx in layers_to_add_to:
            add_test_in_layer(model, layer_idx,
                              "Adding a second node should have also increased the number of out vectors "
                              "by one, and left the numbers of in vectors the same")

    def test_add_node_adds_in_vector_after_added_node(self):
        """
        add node should add an in vector to the layer following the layer with the added node
        """

        def add_test_in_layer(model, layer_idx, msg):
            nr_of_out_vectors_before, nr_of_in_vectors_before = model.layers[layer_idx].weight.size()
            model._add_node(layer_idx)

            if len(model.layers) > layer_idx:
                assert model.layers[layer_idx + 1].weight.size()[1] == nr_of_out_vectors_before + 1, (msg)

        model, layers_to_add_to = model_setup()

        add_test_in_layer(model, layers_to_add_to[0], "add node should add an in vector to the layer following the "
                                                      "layer with the added node, even if done twice in row")

        # testing for random indices:
        for layer_idx in layers_to_add_to:
            add_test_in_layer(model, layer_idx, "add node should add an in vector to the layer following the layer "
                                                "with the added node")

        for layer_idx in layers_to_add_to:
            add_test_in_layer(model, layer_idx, "adding a second node to a layer previously added to should add an in "
                                                "vector to a layer following the layer with the added node")

    def test_add_node_keeps_old_weights(self):
        """
        add node should not change the weights of the old nodes
        """

        model, layers_to_add_to = model_setup()

        for layer_idx in layers_to_add_to:
            nr_of_out_vectors_before, _ = model.layers[layer_idx].weight.size()
            weights_before_add = model.layers[layer_idx].weight

            model._add_node(layer_idx)

            assert model.layers[layer_idx].weight[:-1] == weights_before_add, ("add node should not change the weights "
                                                                               "of the old nodes")

    def test_add_node_changes_voting_layer(self):
        """
        add node should also change the shape of the linear layer responsible to transform the layer output into the 
        voting output
        """
        model, layers_to_add_to = model_setup()

        for layer_idx in layers_to_add_to:
            model._add_node(layer_idx)
            nr_of_vectors_from_layer, _ = model.layers[layer_idx].weight.size()
            nr_of_vectors_into_voting_layer, _ = model.voting_linear_layers[str(layer_idx)].weight.size()

            assert nr_of_vectors_from_layer == nr_of_vectors_into_voting_layer, ("add node should also change the "
                                                                                 "shape of the linear layer "
                                                                                 "responsible to transform the layer "
                                                                                 "output into the voting output")

    def test_add_node_should_still_not_break_forward(self):
        """
        After performing _add_node the functionality of forward should still be intact
        """

        model, layers_to_add_to = model_setup()
        forward_tests = TestAutoDeepLearnerForward()

        for layer_idx in layers_to_add_to:
            model._add_node(layer_idx)
            forward_tests.test_forward_form_single_item_batch()
            forward_tests.test_forward_form_multiple_item_batch()
