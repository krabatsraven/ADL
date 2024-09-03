import torch

from AutoDeepLearner import AutoDeepLearner


class TestAutoDeepLearnerAddNode:
    def test_add_node_adds_node(self):
        """
        add node should add a node to the right layer
        """

        # set up class with a few layers
        feature_count, class_count = torch.randint(10_000, (2,))
        model = AutoDeepLearner(feature_count, class_count)
        nr_of_layers = 100

        for i in range(nr_of_layers):
            model._add_layer()
            
        layers_to_add_to = []

    def test_add_node_keeps_old_weights(self):
        """
        add node should not change the weights of the old nodes
        """
        pass

    def test_add_node_should_still_not_break_forward(self):
        """
        After performing _add_node the functionality of forward should still be intact
        """

        pass
