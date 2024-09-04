import random
from typing import List

import pytest
import torch
from PIL.features import modules
from networkx.utils import np_random_state
from numpy.f2py.crackfortran import previous_context
from sympy.physics.units import amount

from AutoDeepLearner import AutoDeepLearner
from tests.test_AutoDeepLearner.test_forward import TestAutoDeepLearnerForward


class TestAutoDeepLearnerDeleteNode:
    @pytest.fixture(scope="class")
    def feature_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def class_count(self) -> int:
        return random.randint(0, 10_000)

    @pytest.fixture(scope="class")
    def nr_of_layers(self) -> int:
        return 100

    @pytest.fixture(scope="class")
    def max_amount_of_nodes(self):
        return 100

    @pytest.fixture(scope="class")
    def nr_of_nodes_added_in_layer(self, nr_of_layers, max_amount_of_nodes) -> List[int]:
        return [random.randint(1, max_amount_of_nodes) for _ in range(nr_of_layers)]

    @pytest.fixture(autouse=True)
    def model(self, feature_count, class_count, nr_of_layers, nr_of_nodes_added_in_layer) -> AutoDeepLearner:
        model = AutoDeepLearner(feature_count, class_count)

        for idx in range(nr_of_layers):
            model._add_layer()
            for _ in range(nr_of_nodes_added_in_layer[idx]):
                model._add_node(idx)

        yield model

        del model

    @staticmethod
    def previous_context_layers(index, model):
        return model.layers[index].weight.size()[0]

    @staticmethod
    def delete_nodes_with_test(model, previous, test, msg):
        choices_to_delete = [random.randint(0, layer.weight.size()[0]) for layer in model.layers]

        for index, choice in enumerate(choices_to_delete):
            previous_size = previous(index, model)
            model._delete_node(index, choice)
            assert test(index, previous_size, choice, -1), msg

    @staticmethod
    def delete_two_nodes_in_a_row_with_test(model, previous, test, msg):
        choices_to_delete = [random.sample(range(layer.weight.size()[0]), 2) + [index] for index, layer in enumerate(model.layers) if layer.weight.size()[0] > 2]

        for choice_1, choice_2, index in choices_to_delete:
            previous_size = previous(index, model)
            model._delete_node(index, choice_1)
            if choice_2 > choice_1:
                model._delete_node(index, choice_2 - 1)
            else:
                model._delete_node(index, choice_2)

            c_1, c_2 = sorted((choice_1, choice_2))
            assert test(index, previous_size, c_1, c_2), msg

    def test_delete_node_removes_the_node(self, model):
        """
        _delete node should delete the right node in the right layer
        """

        self.delete_nodes_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=lambda index, previous_size, choice_1, choice_2: model.layers[index].weight.size[0] == previous_size - 1,
            msg="_delete node should delete the right node in the right layer"
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=lambda index, previous_size, choice_1, choice_2: model.layers[index].weight.size[0] == previous_size - 2,
            msg="_delete node should delete the right node in the right layer even if done twice in a row"
        )

    def test_delete_node_keeps_the_right_weights(self, model):
        """
        _delete node should not change the weights except for deleting the row of the deleted node
        """
        previous = lambda index, model: model.layers[index].weight

        self.delete_nodes_with_test(
            model=model,
            previous=previous,
            test=lambda index, previous_size, choice_1, choice_2: torch.all(torch.cat((previous_size[:choice_1], previous_size[choice_1:])) == model.layers[index].weight),
            msg="_delete node should not change the weights except for deleting the row of the deleted node"
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=previous,
            test=lambda index, previous_size, choice_1, choice_2: torch.all(torch.cat((previous_size[:choice_1], previous_size[choice_1:choice_2], previous_size[choice_2:])) == model.layers[index].weight),
            msg="_delete node should not change the weights except for deleting the row of the deleted node "
                "even if done twice in a row"
        )

    def test_delete_node_changes_following_layer(self, model):
        """
        _delete node should change the shape of the following layer
        """

        self.delete_nodes_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=lambda index, previous_size, choice_1, choice_2: model.layers[index + 1].weight.size[1] == previous_size - 1 if len(model.layers) > index + 1 else True,
            msg="_delete node should change the shape of the following layer"
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=lambda index, previous_size, choice_1, choice_2: model.layers[index + 1].weight.size[1] == previous_size - 2 if len(model.layers) > index + 1 else True,
            msg="_delete node should change the shape of the following layer even if done twice in a row"
        )

    def test_delete_node_keeps_the_right_weights_in_the_following_layer(self, model):
        """
        _delete node should not change the weights of the following layer except for deleting the column of the deleted node
        """
        previous = lambda index, model: model.layers[index + 1].weight if len(model.layers) > index + 1 else None

        self.delete_nodes_with_test(
            model=model,
            previous=previous,
            test=lambda index, previous_size, choice_1, choice_2: torch.all(torch.cat((previous_size[:, :choice_1], previous_size[:, choice_1:]), dim=1) == model.layers[index].weight) if len(model.layers) > index + 1 else True,
            msg="_delete node should not change the weights of the following layer "
                "except for deleting the column of the deleted node"
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=previous,
            test=lambda index, previous_size, choice_1, choice_2: torch.all(torch.cat((previous_size[:, :choice_1], previous_size[:, choice_1:choice_2], previous_size[:, choice_2:]), dim=1) == model.layers[index].weight) if len(model.layers) > index + 1 else True,
            msg="_delete node should not change the weights of the following layer"
                " except for deleting the column of the deleted node "
                "even if done twice in a row"
        )

    def test_delete_node_changes_voting_layer(self, model):
        """
        _delete node should change the shape of the voting layer
        """
        self.delete_nodes_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=lambda index, previous_size, choice_1, choice_2: model.voting_linear_layers[str(index)].weight.size[1] == previous_size - 1,
            msg="_delete node should change the shape of the voting layer"
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=lambda index, previous_size, choice_1, choice_2: model.voting_linear_layers[str(index)].weight.size[1] == previous_size - 2,
            msg="_delete node should change the shape of the voting layer even if done twice in a row"
        )

    def test_delete_node_keeps_the_right_weights_in_the_voting_layer(self, model):
        """
        _delete node should not change the weights of the voting layer except for deleting the column of the deleted node
        """

        previous = lambda index, model: model.voting_linear_layers[str(index)].weight

        self.delete_nodes_with_test(
            model=model,
            previous=previous,
            test=lambda index, previous_size, choice_1, choice_2: torch.all(torch.cat((previous_size[:, :choice_1], previous_size[:, choice_1:]), dim=1) == model.voting_linear_layers[str(index)].weight),
            msg="_delete node should not change the weights of the voting layer "
                "except for deleting the column of the deleted node"
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=previous,
            test=lambda index, previous_size, choice_1, choice_2: torch.all(torch.cat((previous_size[:, :choice_1], previous_size[:, choice_1:choice_2], previous_size[:, choice_2:]), dim=1) == model.voting_linear_layers[str(index)].weight),
            msg="_delete node should not change the weights of the voting layer "
                "except for deleting the column of the deleted node "
                "even if done twice in a row"
        )

    def test_delete_node_does_not_break_forward_single(self, feature_count, model):
        """
        _delete node should not break the functionality of forward with a batch of one
        """
        self.delete_nodes_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=True,
            msg=""
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=True,
            msg=""
        )

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_single_item_batch(model, feature_count, msg="Single Batch after performing _delete_node: ")

    def test_delete_node_does_not_break_forward_multiple(self, feature_count, model):
        """
        _delete node should not break the functionality of forward with a batch bigger than one row
        """
        self.delete_nodes_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=True,
            msg=""
        )

        self.delete_two_nodes_in_a_row_with_test(
            model=model,
            previous=self.previous_context_layers,
            test=True,
            msg=""
        )

        forward_tests = TestAutoDeepLearnerForward()
        forward_tests.test_forward_form_multiple_item_batch(model, feature_count, batch_size=1000, msg="Multiple Batch after performing _delete_node: ")