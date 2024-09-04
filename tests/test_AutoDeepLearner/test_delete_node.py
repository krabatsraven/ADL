import pytest

class TestAutoDeepLearnerDeleteNode:
    def test_delete_node_removes_the_node(self):
        """
        _delete node should delete the right node in the right layer
        """
        pass

    def test_delete_node_keeps_the_right_weights(self):
        """
        _delete node should not change the weights except for deleting the row of the deleted node
        """
        pass

    def test_delete_node_changes_following_layer(self):
        """
        _delete node should change the shape of the following layer
        """
        pass

    def test_delete_node_keeps_the_right_weights_in_the_following_layer(self):
        """
        _delete node should not change the weights of the following layer except for deleting the column of the deleted node
        """
        pass

    def test_delete_node_changes_voting_layer(self):
        """
        _delete node should change the shape of the voting layer
        """
        pass

    def test_delete_node_keeps_the_right_weights_in_the_voting_layer(self):
        """
        _delete node should not change the weights of the voting layer except for deleting the column of the deleted node
        """
        pass

    def test_delete_node_does_not_break_forward_single(self):
        """
        _delete node should not break the functionality of forward with a batch of one
        """
        pass

    def test_delete_node_does_not_break_forward_multiple(self):
        """
        _delete node should not break the functionality of forward with a batch bigger than one row
        """
        pass