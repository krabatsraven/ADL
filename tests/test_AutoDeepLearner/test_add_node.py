import random
import pytest
from typing import List, Tuple

import torch

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


# def model_setup() -> Tuple[AutoDeepLearner, List[int]]:
#     # set up class with a few layers
#     feature_count, class_count = torch.randint(10_000, (2,))
#     model = AutoDeepLearner(feature_count, class_count)
#     nr_of_layers = 100
# 
#     for i in range(nr_of_layers):
#         model._add_layer()
# 
#     # draw a random amount of indices to add a node too
#     layers_to_add_to = [random.randint(0, nr_of_layers) for _ in range(random.randint(0, nr_of_layers))]
# 
#     return model, layers_to_add_to


class TestAutoDeepLearnerAddNode:

    @pytest.fixture(scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def nr_of_layers(self) -> int:
        return 100

    @pytest.fixture(autouse=True)
    def model(self, feature_count, class_count, nr_of_layers) -> AutoDeepLearner:
        model = AutoDeepLearner(feature_count, class_count)

        for i in range(nr_of_layers):
            model._add_layer()

        yield model

        del model

    @pytest.fixture(scope="class", autouse=True)
    def layers_to_add_to(self, nr_of_layers: int) -> List[int]:
        return [random.randint(0, nr_of_layers) for _ in range(random.randint(0, nr_of_layers))]

    def test_add_node_adds_node(self, model, layers_to_add_to):
        """
        add node should add a node to the right layer
        """

        def add_test_in_layer(model, layer_idx, msg):
            nr_of_out_vectors_before, nr_of_in_vectors_before = model.layers[layer_idx].weight.size()
            model._add_node(layer_idx)

            assert model.layers[layer_idx].weight.size()[0] == nr_of_out_vectors_before + 1, (msg)
            assert model.layers[layer_idx].weight.size()[1] == nr_of_in_vectors_before, (msg)

        # testing for random indices:
        for layer_idx in layers_to_add_to:
            add_test_in_layer(
                model,
                layer_idx,
                ("Adding a node should have increased the number of out vectors by one, "
                 "and left the numbers of in vectors the same"))

        # testing for random indices:
        for layer_idx in layers_to_add_to:
            add_test_in_layer(
                model,
                layer_idx,
                ("Adding a second node should have also increased the number of out vectors by one, "
                 "and left the numbers of in vectors the same"))

    def test_add_node_adds_in_vector_after_added_node(self, model, layers_to_add_to):
        """
        add node should add an in vector to the layer following the layer with the added node
        """

        def add_test_in_layer(model, layer_idx, msg):
            nr_of_out_vectors_before, nr_of_in_vectors_before = model.layers[layer_idx].weight.size()
            model._add_node(layer_idx)

            if len(model.layers) - 1 > layer_idx:
                assert model.layers[layer_idx + 1].weight.size()[1] == nr_of_out_vectors_before + 1, (msg)

        add_test_in_layer(
            model,
            layers_to_add_to[0],
            ("add node should add an in vector to the layer following the layer with the added node, "
             "even if done twice in row"))

        # testing for random indices:
        for layer_idx in layers_to_add_to:
            add_test_in_layer(
                model,
                layer_idx,
                "add node should add an in vector to the layer following the layer with the added node")

        for layer_idx in layers_to_add_to:
            add_test_in_layer(
                model,
                layer_idx,
                ("adding a second node to a layer previously added to "
                "should add an in vector to a layer following the layer with the added node"))

    def test_add_node_keeps_old_weights(self, model, layers_to_add_to):
        """
        add node should not change the weights of the old nodes
        """

        for layer_idx in layers_to_add_to:
            nr_of_out_vectors_before, _ = model.layers[layer_idx].weight.size()
            weights_before_add = model.layers[layer_idx].weight

            model._add_node(layer_idx)

            assert (torch.all(model.layers[layer_idx].weight[:-1] == weights_before_add),
                    (f"add node should not change the weights "
                     f"of the old nodes: {model.layers[layer_idx].weight[:-1] == weights_before_add}"))

    def test_add_node_changes_voting_layer(self, model, layers_to_add_to):
        """
        add node should also change the shape of the linear layer responsible to transform the layer output into the 
        voting output
        """

        for layer_idx in layers_to_add_to:
            model._add_node(layer_idx)
            nr_of_vectors_from_layer, _ = model.layers[layer_idx].weight.size()
            _, nr_of_vectors_into_voting_layer = model.voting_linear_layers[str(layer_idx)].weight.size()

            assert (nr_of_vectors_from_layer == nr_of_vectors_into_voting_layer,
                    ("add node should also change the shape of the linear layer "
                    "responsible to transform the layer output into the voting output"))

    def test_add_node_should_still_not_break_forward(self, model, feature_count, layers_to_add_to):
        """
        After performing _add_node the functionality of forward should still be intact
        """

        for layer_idx in layers_to_add_to:
            model._add_node(layer_idx)

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(
            model,
            feature_count,
            msg="After performing _add_node: "
        )

    def test_add_node_should_still_not_break_forward_multiple(self, model, feature_count, layers_to_add_to):
        """
        After performing _add_node the functionality of forward should still be intact
        """

        for layer_idx in layers_to_add_to:
            model._add_node(layer_idx)

        forward_tests = TestAutoDeepLearnerForward()
        # setting batch size too big will break the memory
        forward_tests.test_forward_form_multiple_item_batch(
            model,
            feature_count,
            batch_size=1000,
            msg="After performing _add_node "
        )

    def test_add_node_raises_on_negative_index(self, model):
        layer_index = -1
        error_str = (f"cannot add a node to layer with the index {layer_index}, "
                     f"as it is not in the range [0, amount of layers in model]")
        with pytest.raises(Exception) as exec_info:
            model._add_node(layer_index)

        assert str(exec_info.value) == error_str, "negative indices should raise an exception"

    def test_add_node_raises_on_to_big_index(self, model):
        layer_index = len(model.layers)
        error_str = (f"cannot add a node to layer with the index {layer_index}, "
                     f"as it is not in the range [0, amount of layers in model]")
        with pytest.raises(Exception) as exec_info:
            model._add_node(layer_index)

        assert str(exec_info.value) == error_str, ("indices bigger than or equal to "
                                                   "the length of the list of layers should raise an exception")

    def test_add_node_raises_on_no_voting_linear_layer(self, model, nr_of_layers):
        layer_index = random.randint(0, nr_of_layers - 1)
        model.voting_linear_layers.pop(str(layer_index))
        error_str = (f"cannot add a node to layer with the index {layer_index}, "
                     f"as it is not a layer that will projected onto a vote")
        with pytest.raises(Exception) as exec_info:
            model._add_node(layer_index)

        assert str(exec_info.value) == error_str, \
            "a layer index without a voting linear layer should raise an exception"
