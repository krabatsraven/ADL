from typing import List, Dict, Optional

import numpy as np
import torch
from numpy._typing import NDArray
from sympy.codegen.fnodes import reshape
from torch import nn, dtype


class AutoDeepLearner(nn.Module):

    def __init__(self,
                 nr_of_features: int,
                 nr_of_classes: int,
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)

        self.input_size: int = nr_of_features
        self.output_size: int = nr_of_classes

        # assumes that at least a single concept is to be learned in a data stream
        # list of all dynamic layers, starting with a single layer in the shape (in, 1)
        first_hidden_layer = nn.Linear(nr_of_features, 1, dtype=torch.float)
        nn.init.xavier_normal_(first_hidden_layer.weight)
        self.layers: nn.ModuleList = nn.ModuleList([first_hidden_layer])

        # list of all linear layers for the voting part, starts with a single layer in the shape (1, out)
        # contains only the indices of self.layers that are eligible to vote
        first_output_layer = nn.Linear(1, nr_of_classes, dtype=torch.float)
        nn.init.xavier_normal_(first_output_layer.weight)
        self.voting_linear_layers: nn.ModuleDict = nn.ModuleDict({'0': first_output_layer})

        # all voting weights should always be normalised,
        # and only contain the indices of self.layers that eligible to vote
        self.voting_weights: Dict[int, float] = {0: 1.0}

        # for the adjustment of the weights in the optimizer it is necessary to have the results of the single voting layers
        self.layer_result_keys: Optional[NDArray[np.int_]] = None
        self.layer_results: Optional[torch.Tensor] = None

        # for the adjustment of the weights in the optimizer it is necessary to keep track of a correction_factor for each layer
        self.initial_weight_correction_factor: float = 0.5
        self.weight_correction_factor: Dict[int, float] = {0: self.initial_weight_correction_factor}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns the classification from self.output_size many classes stemming from the data input x
        alternative: defines the forward pass through the network
        compare section 4.1 equation 1 of the paper
        :param x: batch of data to be classified with self.input_size many features 
        :return: classification tensor that contains the index of the class choosen from the multiclass problem
        """

        # check if x is of right dimension (sanity check)
        if len(x.size()) > 1:
            assert x.size()[1] == self.input_size, \
                f"Given batch of data has {x.size()[1]} many features, expected where {self.input_size}"
        else:
            assert x.size()[0] == self.input_size,\
                f"Given batch of data has {x.size()[0]} many features, expected where {self.input_size}"

        # calculate all h^{l} = \sigma(W^{l} h^{(l-1)} + b^{l}), h^{(0)} = x
        hidden_layers: List[torch.Tensor] = [x := nn.Sigmoid()(layer(x)) for layer in self.layers]

        # calculate all y^i = s.max(W_{s_l}h^{l} + b_{s_l})
        # that are not currently pruned
        self.layer_result_keys = np.array(list(self.voting_weights.keys()))
        self.layer_results = torch.stack([nn.Softmax()(self.voting_linear_layers[str(i)](hidden_layers[i]))
                                          for i in self.layer_result_keys])

        # add n empty dimensions at the end of betas dimensionality to allow for multiplying with the layer results:
        # e.g.: beta.size = (layers) -> beta.size = (layers, 1, 1) or beta.size = (layers, 1) if batch size is 1
        # n is 2 for a batch size greater than 1 else it is 1
        betas = torch.tensor([beta for beta in self.voting_weights.values()])[(..., ) + (None,) * (len(self.layer_results.size()) - 1)]

        # calculated total voted/weighted class probability
        total_weighted_class_probability = torch.mul(self.layer_results, betas).sum(dim=0)

        return total_weighted_class_probability

    def _add_layer(self) -> None:
        """
        adds a new untrained layer at the end of the first network
        """

        previous_layer_output_size, previous_layer_input_size = self.layers[-1].weight.size()

        # new layers are initialised with one node
        nr_of_out_nodes = 1
        new_layer = nn.Linear(previous_layer_output_size, nr_of_out_nodes)
        nn.init.xavier_normal_(new_layer.weight)
        self.layers.append(new_layer)

        idx_of_new_layer = len(self.layers) - 1
        new_output_layer = nn.Linear(nr_of_out_nodes, self.output_size)
        nn.init.xavier_normal_(new_output_layer.weight)
        self.voting_linear_layers[str(idx_of_new_layer)] = new_output_layer
        self.voting_weights[idx_of_new_layer] = 0
        self.weight_correction_factor[idx_of_new_layer] = self.initial_weight_correction_factor

    def _prune_layer_by_vote_removal(self, layer_index: int) -> None:
        """
        removes the layer with the given index from the voting process
        :param layer_index: index of the layer to be removed, inside the list of layers
        """

        # check whether layer exists and can vote (sanity check)
        assert 0 <= layer_index < len(self.layers), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not in the range [0, amount of layers in model]")
        assert str(layer_index) in self.voting_linear_layers.keys(), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not a layer that will projected onto a vote")
        assert layer_index in self.voting_weights.keys(), \
            (f"cannot remove the layer with the index {layer_index}, "
            f"as it is not a layer that can vote because it has no voting weight")
        assert layer_index in self.weight_correction_factor, \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not a layer that has no weight correction factor")
        assert any(True for key, value in self.voting_weights.items() if key != layer_index and value != 0), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is the last layer with a non zero voting weight")

        # remove layer from self.voting_linear_layers, and thereby from voting
        self.voting_linear_layers.pop(str(layer_index))
        # remove the weight of the layer from self.voting_weights
        self.voting_weights.pop(layer_index)
        # remove the correction_factor from self.weight_correction_factor
        self.weight_correction_factor.pop(layer_index)
        # and re-normalize the voting weights?
        self._normalise_voting_weights()

    def _normalise_voting_weights(self) -> None:
        voting_weights_keys_vector: NDArray[int] = np.fromiter(self.voting_weights.keys(), dtype=int)
        voting_weights_values_vector: NDArray[float] = np.fromiter(self.voting_weights.values(), dtype=float)
        norm_of_voting_weights: np.floating = np.linalg.norm(voting_weights_values_vector)
        assert norm_of_voting_weights != 0, \
            "The voting weights vector has a length of zero and cannot be normalised"
        voting_weights_values_vector /= norm_of_voting_weights
        for index, key in np.ndenumerate(voting_weights_keys_vector):
            self.voting_weights[key] = float(voting_weights_values_vector[index])

    def _add_node(self, layer_index: int) -> None:
        """
        adds a new node to the layer l_{layer_index} to the bottom of the layer/ to the bottom of the matrix

        :param layer_index: the index of the layer in the list of layers
        """

        # check whether layer exists (sanity check)
        assert 0 <= layer_index < len(self.layers),\
            (f"cannot add a node to layer with the index {layer_index}, "
             f"as it is not in the range [0, amount of layers in model]")
        assert str(layer_index) in self.voting_linear_layers.keys(),\
            (f"cannot add a node to layer with the index {layer_index}, "
             f"as it is not a layer that will projected onto a vote")

        # find layer
        layer_to_add_to = self.layers[layer_index]

        # generate layer
        out_before, in_before = layer_to_add_to.weight.size()
        new_layer = nn.Linear(in_before, out_before + 1)

        # the new weight is set by Xavier Initialisation
        nn.init.xavier_normal_(new_layer.weight)

        # with the same weights and one additional one
        new_layer.weight = nn.parameter.Parameter(torch.cat((layer_to_add_to.weight, new_layer.weight[0:1])))
        new_layer.bias = nn.parameter.Parameter(torch.cat((layer_to_add_to.bias, new_layer.bias[0:1])))

        # change layer
        self.layers[layer_index] = new_layer

        # change following layer
        if layer_index < len(self.layers) - 1:
            old_following_layer = self.layers[layer_index + 1]
            anount_out_vectors, amount_in_vectors = old_following_layer.weight.size()
            new_following_layer = nn.Linear(amount_in_vectors + 1, anount_out_vectors)
            nn.init.xavier_normal_(new_following_layer.weight)
            new_following_layer.weight = nn.parameter.Parameter(
                torch.cat((old_following_layer.weight, new_following_layer.weight[:, 0:1]), dim=1))
            self.layers[layer_index + 1] = new_following_layer

        # change the shape of Ws^l
        # (the linear layer that takes the outputs of the hidden layer for the voting function)
        old_voting_layer = self.voting_linear_layers[str(layer_index)]
        new_voting_layer = nn.Linear(out_before + 1, self.output_size)

        nn.init.xavier_normal_(new_voting_layer.weight)

        new_voting_layer.weight = nn.parameter.Parameter(
            torch.cat((old_voting_layer.weight, new_voting_layer.weight[:, 0:1]), dim=1))
        # has the same amount of outputs -> has the same amount of biases
        new_voting_layer.bias = nn.parameter.Parameter(old_voting_layer.bias)

        # change voting layer
        self.voting_linear_layers[str(layer_index)] = new_voting_layer

    def _delete_node(self, layer_index: int, node_index: int) -> None:
        """
        deletes from layer l_{layer_index} the node with index node_index
        implementation:
            it will create a new layer with an out reduced by 1,
            the weights will be the same as the old ones, but missing the row node_index
            the following layer, as well as the voting layer, will also be changed, now of the form (old_out) x (old_in - 1)

        :param layer_index: the index of the layer in the list of layers to be changed
        :param node_index: the index of node to be deleted
        """

        def create_new_layer_with_deleted_node_index(old_following_layer: nn.Module, node_index: int) -> nn.Linear:
            old_following_out, old_following_in = old_following_layer.weight.size()
            # create new nn.linear(in=old_in - 1, out=old_out)
            new_following_layer = nn.Linear(old_following_in - 1, old_following_out)
            # set weights of new layer to the one of the original missing the node_index row (zero indexed)
            new_following_layer.weight = nn.parameter.Parameter(torch.cat((old_following_layer.weight[:, :node_index], old_following_layer.weight[:, node_index + 1:]), dim=1))
            return new_following_layer

        # check whether layer exists (sanity check)
        assert 0 <= layer_index < len(self.layers),\
            (f"cannot remove a node from the layer with the index {layer_index},"
             f" as it is not in the range [0, amount of layers in model]")
        assert str(layer_index) in self.voting_linear_layers.keys(),\
            (f"cannot remove a node from the layer with the index {layer_index}, "
             f"as it is not a layer that will projected onto a vote")

        old_layer = self.layers[layer_index]

        # create new nn.linear(in=old_in, out=old_out - 1)
        old_out, old_in = old_layer.weight.size()

        # check whether layer has node to delete (sanity check)
        assert 0 <= node_index < old_out,\
            (f"cannot remove the node with index {node_index} from the layer with the index {layer_index}, "
             f"as it has no node with index {node_index}")

        new_layer = nn.Linear(old_in, old_out - 1)
        # set weights of new layer to the one of the original missing the node_index row (zero indexed)
        new_layer.weight = nn.parameter.Parameter(torch.cat((old_layer.weight[:node_index], old_layer.weight[node_index + 1:]), dim=0))
        # set bias of new layer to the one of the original missing the node_index row (zero indexed)
        new_layer.bias = nn.parameter.Parameter(torch.cat((old_layer.bias[:node_index], old_layer.bias[node_index + 1:]), dim=0))
        self.layers[layer_index] = new_layer

        # change the next layer after layer_index
        if layer_index < len(self.layers) - 1:
            self.layers[layer_index + 1] = create_new_layer_with_deleted_node_index(self.layers[layer_index + 1], node_index)

        # change voting layer the same as the layer next_layer + 1
        self.voting_linear_layers[str(layer_index)] = create_new_layer_with_deleted_node_index(self.voting_linear_layers[str(layer_index)], node_index)
