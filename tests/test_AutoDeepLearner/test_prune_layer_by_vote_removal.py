class TestPruneLayerByVoteRemoval:
    def test_prune_layer_by_vote_removal_removes_vote(self):
        """
        _prune_layer_by_vote_removal should remove the layer from model.voting_linear_layers
        """
        pass

    def test_prune_layer_by_vote_removal_does_not_remove_layer(self):
        """
        _prune_layer_by_vote_removal does not touch the layer in self.layers
        """
        pass

    def test_prune_layer_by_vote_removal_removes_voting_weight(self):
        """
        _prune_layer_by_vote_removal should remove the layer from the voting weights
        """
        pass

    def test_after_prune_layer_by_vote_removal_voting_weights_are_normalized(self):
        """
        _prune_layer_by_vote_removal should leave the voting weights normalized
        """
        pass

    def test_prune_layer_by_vote_removal_does_not_break_forward_single_item(self):
        """
        _prune_layer_by_vote_removal should not affect the functionality of the forward pass on single item batches
        """
        pass

    def test_prune_layer_by_vote_removal_does_not_break_backward_multiple_item(self):
        """
        _prune_layer_by_vote_removal should not affect the functionality of the backward pass on multiple item batches
        """
        pass