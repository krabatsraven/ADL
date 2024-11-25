from typing import List, Dict, Optional

import numpy as np
import torch
from torch import nn


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

        # lowest possible value that voting weights and voting weight correction factors can assume
        self._epsilon: float = 10 ** -6

        # assumes that at least a single concept is to be learned in a data stream
        # list of all dynamic layers, starting with a single layer in the shape (in, 1)
        self.layers: nn.ModuleList = nn.ModuleList()

        first_hidden_layer = nn.Linear(nr_of_features, 1, dtype=torch.float)
        nn.init.xavier_normal_(first_hidden_layer.weight)
        self.layers.append(first_hidden_layer)

        # list of all linear layers for the voting part, starts with a single layer in the shape (1, out)
        # contains only the indices of self.layers that are eligible to vote
        first_output_layer = nn.Linear(1, nr_of_classes, dtype=torch.float)
        nn.init.xavier_normal_(first_output_layer.weight)
        self.voting_linear_layers: nn.ModuleDict = nn.ModuleDict({f'output layer {0}': first_output_layer})

        # voting weights:
        # are dynamically adjusted
        # upper and lower boarder for the domain of the voting weights beta(l) for layer l
        # lower_voting_weigth_boarder <= beta(l) <= upper_voting_weight_boarder for all l
        self.lower_voting_weigth_boarder: float = self._epsilon
        self.upper_voting_weight_boarder: float = 1

        # all voting weights should always be normalised,
        # and only contain the indices of self.layers that eligible to vote
        self.weight_initializiation_value: float = 1.0
        self.voting_weights: nn.ParameterDict = nn.ParameterDict({'0': self.weight_initializiation_value})

        # for the adjustment of the weights in the optimizer
        # it is necessary to have the results of the single voting layers
        self.layer_result_keys: Optional[torch.Tensor] = None
        self.layer_results: Optional[torch.Tensor] = None

        # voting weights correction factors:
        # voting weights correction factors are dynamically adjusted
        # for the adjustment of the weights in the optimizer

        # upper and lower boarder for the domain of the voting weight correction factors p(l) for layer l:
        # self.lower_weigth_correction_factor_boarder <= p(l) <= self.upper_weigth_correction_factor_boarder for all l
        self.lower_weigth_correction_factor_boarder: float = self._epsilon
        self.upper_weigth_correction_factor_boarder: float = 1

        # it is necessary to keep track of a correction_factor for each layer
        self.weight_correction_factor_initialization_value: float = 1
        self.weight_correction_factor: nn.ParameterDict = nn.ParameterDict(
            {'0': self.weight_correction_factor_initialization_value}
        )

        # the accuracy matrix is used for univariant drift detection
        # therefore it is necessary to save the last prediction
        self.last_prediction: Optional[torch.Tensor] = None

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
            assert x.size()[0] == self.input_size, \
                f"Given batch of data has {x.size()[0]} many features, expected where {self.input_size}"

        # if exclude_layer_indicies_in_training is not None:
        #     # check whether the indicies provided to be excluded from training are valid:
        #     # todo: test this part
        #     assert all((isinstance(elem, int) for elem in exclude_layer_indicies_in_training)), \
        #         (f"the provided list of indicies to exclude in training does not consist of only ints: "
        #          f"exclude_layer_indicies_in_training={exclude_layer_indicies_in_training}")
        #     assert all((0 <= index < len(self.layers) for index in exclude_layer_indicies_in_training)), \
        #         ("The provided list of indicies to exclude in training does not consist only of indicies,"
        #          "that are valid for the list of hidden layers"
        #          f"exclude_layer_indicies_in_training={exclude_layer_indicies_in_training}")
        #     assert all((self.output_layer_with_index_exists(index) for index in exclude_layer_indicies_in_training)), \
        #         (f"Not all provided indicies are valid output layer indices, "
        #          f"exclude_layer_indicies_in_training={exclude_layer_indicies_in_training}")
        #     # assert all(self.layers[index].requires_grad for index in exclude_layer_indicies_in_training), \
        #     #     (f"Not all layers that where listed to be excluded from training are enabled,"
        #     #      f"exclude_layer_indicies_in_training={exclude_layer_indicies_in_training}")
        #     # assert all(self.get_output_layer(index).requires_grad for index in exclude_layer_indicies_in_training),\
        #     #     (f"Not all output layers that where listed to be excluded from training are enabled,"
        #     #      f"exclude_layer_indicies_in_training={exclude_layer_indicies_in_training}")
        #
        #     # set requires grad to False
        #     for excluded_index in exclude_layer_indicies_in_training:
        #         self.layers[excluded_index].requires_grad_(requires_grad=False)
        #         self.get_output_layer(excluded_index).requires_grad_(requires_grad=False)

        # calculate all h^{l} = \sigma(W^{l} h^{(l-1)} + b^{l}), h^{(0)} = x
        hidden_layers: List[torch.Tensor] = [x := nn.Sigmoid()(layer(x)) for layer in self.layers]

        # calculate all y^i = s.max(W_{s_l}h^{l} + b_{s_l})
        # that are not currently pruned
        self.layer_result_keys = self.get_keys_of_active_layers().numpy()
        layer_results = torch.stack([nn.Softmax(dim=-1)(self.get_output_layer(i)(hidden_layers[i]))
                                          for i in self.layer_result_keys])

        # todo: change to support shape=(x,N,M) where N = nr of hidden layers, M = nr of classes and x = nr of instances
        # todo: currently it is assumed that x is 1, but:
        # todo: currently expected a bug if more than one instances is fed into the network at once
        self.layer_results = layer_results.flatten(end_dim=1)

        # add n empty dimensions at the end of betas dimensionality to allow for multiplying with the layer results:
        # e.g.: beta.size = (layers) -> beta.size = (layers, 1, 1) or beta.size = (layers, 1) if batch size is 1
        # n is 2 for a batch size greater than 1 else it is 1
        betas = self.get_voting_weight_values()[(...,) + (None,) * (len(layer_results.size()) - 1)]

        # calculated total voted/weighted class probability
        total_weighted_class_probability = torch.mul(layer_results, betas).sum(dim=0)
        self.last_prediction = torch.argmax(total_weighted_class_probability)

        # if exclude_layer_indicies_in_training is not None:
        #     # set requires grad to True/ re-enable them
        #     for excluded_index in exclude_layer_indicies_in_training:
        #         self.layers[excluded_index].requires_grad_(requires_grad=True)
        #         self.get_output_layer(excluded_index).requires_grad_(requires_grad=True)

        return total_weighted_class_probability

    def _disable_layers_for_training(self, layer_indicies: List[int]):
        assert all((isinstance(elem, int) for elem in layer_indicies)), \
            (f"the provided list of indicies to exclude in training does not consist of only ints: "
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert all((0 <= index < len(self.layers) for index in layer_indicies)), \
            ("The provided list of indicies to exclude in training does not consist only of indicies,"
             "that are valid for the list of hidden layers"
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert all((self.output_layer_with_index_exists(index) for index in layer_indicies)), \
            (f"Not all provided indicies are valid output layer indices, "
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert all(
            (self.layers[index].weight.requires_grad and self.layers[index].bias.requires_grad)
            for index in layer_indicies), \
            (f"Not all layers that where listed to be excluded from training are enabled,"
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert all(
            (self.get_output_layer(index).weight.requires_grad and self.get_output_layer(index).bias.requires_grad)
            for index in layer_indicies), \
            (f"Not all output layers that where listed to be excluded from training are enabled,"
             f"exclude_layer_indicies_in_training={layer_indicies}")

        # set requires grad to False
        for excluded_index in layer_indicies:
            self.layers[excluded_index].requires_grad_(requires_grad=False)
            self.get_output_layer(excluded_index).requires_grad_(requires_grad=False)

    def _enable_layers_for_training(self, layer_indicies: List[int]):
        assert all((isinstance(elem, int) for elem in layer_indicies)), \
            (f"the provided list of indicies to exclude in training does not consist of only ints: "
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert all((0 <= index < len(self.layers) for index in layer_indicies)), \
            ("The provided list of indicies to exclude in training does not consist only of indicies,"
             "that are valid for the list of hidden layers"
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert all((self.output_layer_with_index_exists(index) for index in layer_indicies)), \
            (f"Not all provided indicies are valid output layer indices, "
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert not any(
            (self.layers[index].weight.requires_grad or self.layers[index].bias.requires_grad)
            for index in layer_indicies), \
            (f"Not all layers that where listed to be enabled for training were disabled,"
             f"exclude_layer_indicies_in_training={layer_indicies}")
        assert not any(
            (self.get_output_layer(index).weight.requires_grad or self.get_output_layer(index).bias.requires_grad)
            for index in layer_indicies), \
            (f"Not all output layers that where listed to be enabled for training are disabled,"
             f"exclude_layer_indicies_in_training={layer_indicies}")

        # set requires grad to True
        for excluded_index in layer_indicies:
            self.layers[excluded_index].requires_grad_(requires_grad=True)
            self.get_output_layer(excluded_index).requires_grad_(requires_grad=True)

    def _add_layer(self) -> None:
        """
        adds a new untrained layer at the end of the first network
        """

        previous_layer_output_size, previous_layer_input_size = self.layers[-1].weight.size()

        # new layers are initialised with one node
        nr_of_out_nodes = 1
        new_layer = nn.Linear(previous_layer_output_size, nr_of_out_nodes)

        nn.init.xavier_normal_(new_layer.weight)

        # add new hidden layer to layer list
        idx_of_new_layer = len(self.layers)
        self.layers.append(new_layer)

        # add new output layer
        new_output_layer = nn.Linear(nr_of_out_nodes, self.output_size)
        nn.init.xavier_normal_(new_output_layer.weight)
        self.__set_output_layer(idx_of_new_layer, new_output_layer)

        # add new weight
        self.__set_voting_weight(idx_of_new_layer, self.weight_initializiation_value)
        self.__set_weight_correction_factor(idx_of_new_layer, self.weight_correction_factor_initialization_value)

    def _prune_layer_by_vote_removal(self, layer_index: int) -> None:
        """
        removes the layer with the given index from the voting process
        :param layer_index: index of the layer to be removed, inside the list of layers
        """

        # check whether layer exists and can vote (sanity check)
        assert 0 <= layer_index < len(self.layers), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not in the range [0, amount of layers in model]")
        assert self.output_layer_with_index_exists(layer_index), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not a layer that will projected onto a vote")
        assert self.voting_weight_with_index_exists(layer_index), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not a layer that can vote because it has no voting weight")
        assert self.weight_correction_factor_with_index_exists(layer_index), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is not a layer that has no weight correction factor")
        assert (any((value != 0) for value in self.get_voting_weight_values()[:layer_index]) 
                or any((value != 0) for value in self.get_voting_weight_values()[layer_index + 1:])), \
            (f"cannot remove the layer with the index {layer_index}, "
             f"as it is the last layer with a non zero voting weight")

        # remove layer from self.voting_linear_layers, and thereby from voting
        self.__pop_output_layer(layer_index)
        # remove the weight of the layer from self.voting_weights
        self.__pop_voting_weight(layer_index)
        # remove the correction_factor from self.weight_correction_factor
        self.__pop_weight_correction_factor(layer_index)
        # and re-normalize the voting weights?
        self._normalise_voting_weights()

    def _normalise_voting_weights(self) -> None:
        voting_weights_keys_vector = self.get_keys_of_active_layers()
        voting_weights_values_vector = self.get_voting_weight_values()

        norm_of_voting_weights = torch.linalg.norm(voting_weights_values_vector, ord=2, dim=0)
        assert norm_of_voting_weights != 0, \
            "The voting weights vector has a length of zero and cannot be normalised"

        voting_weights_values_vector = nn.functional.normalize(voting_weights_values_vector, p=1, dim=0)
        for index, key in np.ndenumerate(voting_weights_keys_vector):
            self.__set_voting_weight(int(key), float(voting_weights_values_vector[index]))

    def _add_node(self, layer_index: int) -> None:
        """
        adds a new node to the layer l_{layer_index} to the bottom of the layer/ to the bottom of the matrix

        :param layer_index: the index of the layer in the list of layers
        """

        # check whether layer exists (sanity check)
        assert 0 <= layer_index < len(self.layers), \
            (f"cannot add a node to layer with the index {layer_index}, "
             f"as it is not in the range [0, amount of layers in model]")
        assert self.output_layer_with_index_exists(layer_index), \
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
            amount_out_vectors, amount_in_vectors = old_following_layer.weight.size()
            new_following_layer = nn.Linear(amount_in_vectors + 1, amount_out_vectors)
            nn.init.xavier_normal_(new_following_layer.weight)
            new_following_layer.weight = nn.parameter.Parameter(
                torch.cat((old_following_layer.weight, new_following_layer.weight[:, 0:1]), dim=1))
            self.layers[layer_index + 1] = new_following_layer

        # change the shape of Ws^l
        # (the linear layer that takes the outputs of the hidden layer for the voting function)
        old_voting_layer = self.get_output_layer(layer_index)
        new_voting_layer = nn.Linear(out_before + 1, self.output_size)

        nn.init.xavier_normal_(new_voting_layer.weight)

        new_voting_layer.weight = nn.parameter.Parameter(
            torch.cat((old_voting_layer.weight, new_voting_layer.weight[:, 0:1]), dim=1))
        # has the same amount of outputs -> has the same amount of biases
        new_voting_layer.bias = nn.parameter.Parameter(old_voting_layer.bias)

        # change voting layer
        self.__set_output_layer(layer_index, new_voting_layer)

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
            new_following_layer.weight = nn.parameter.Parameter(
                torch.cat((old_following_layer.weight[:, :node_index], old_following_layer.weight[:, node_index + 1:]),
                          dim=1))
            return new_following_layer

        # check whether layer exists (sanity check)
        assert 0 <= layer_index < len(self.layers), \
            (f"cannot remove a node from the layer with the index {layer_index},"
             f" as it is not in the range [0, amount of layers in model]")
        assert self.output_layer_with_index_exists(layer_index), \
            (f"cannot remove a node from the layer with the index {layer_index}, "
             f"as it is not a layer that will projected onto a vote")

        old_layer = self.layers[layer_index]

        # create new nn.linear(in=old_in, out=old_out - 1)
        old_out, old_in = old_layer.weight.size()

        # check whether layer has node to delete (sanity check)
        assert 0 <= node_index < old_out, \
            (f"cannot remove the node with index {node_index} from the layer with the index {layer_index}, "
             f"as it has no node with index {node_index}")

        new_layer = nn.Linear(old_in, old_out - 1)
        # set weights of new layer to the one of the original missing the node_index row (zero indexed)
        new_layer.weight = nn.parameter.Parameter(
            torch.cat((old_layer.weight[:node_index], old_layer.weight[node_index + 1:]), dim=0))
        # set bias of new layer to the one of the original missing the node_index row (zero indexed)
        new_layer.bias = nn.parameter.Parameter(
            torch.cat((old_layer.bias[:node_index], old_layer.bias[node_index + 1:]), dim=0))
        self.layers[layer_index] = new_layer

        # change the next layer after layer_index
        if layer_index < len(self.layers) - 1:
            self.layers[layer_index + 1] = create_new_layer_with_deleted_node_index(self.layers[layer_index + 1],
                                                                                    node_index)

        # change voting layer the same as the layer next_layer + 1
        new_output_layer = create_new_layer_with_deleted_node_index(
            self.get_output_layer(layer_index), node_index)

        self.__set_output_layer(layer_index, new_output_layer)

    # getter and setter methods for module and parameter dicts:
    # --------------------------------------------------------
    def get_output_layer(self, layer_index: int) -> nn.Module:
        """
        :returns the output layer of the hidden layer at given index
        :param layer_index: the index of the hidden layer in the self.layers list
        :return: linear layer of dim (out of hidden layer, number of classes)
        """
        return self.voting_linear_layers[f'output layer {layer_index}']

    def __set_output_layer(self, layer_index: int, new_output_layer) -> None:
        self.voting_linear_layers[f'output layer {layer_index}'] = new_output_layer

    def __pop_output_layer(self, layer_index: int) -> None:
        self.voting_linear_layers.pop(f'output layer {layer_index}')

    def output_layer_with_index_exists(self, layer_index: int) -> bool:
        """
        :returns whether the layer with the given index has an output layer associated with it
        """
        return f'output layer {layer_index}' in self.voting_linear_layers.keys()

    def get_voting_weight(self, layer_index: int) -> float:
        """
        returns the factor with which the result of the l-th output layer is weighted in the final result
        :param layer_index: the index of the hidden layer in the self.layers list
        :return: factor between 0 and 1, weights the result of the i-th output layer
        """

        return self.voting_weights[str(int(layer_index))]

    def __set_voting_weight(self, layer_index: int, new_weight: float) -> None:
        assert isinstance(layer_index, int)
        self.voting_weights[str(layer_index)] = new_weight

    def __pop_voting_weight(self, layer_index: int) -> float:
        return self.voting_weights.pop(str(int(layer_index)))

    def voting_weight_with_index_exists(self, layer_index: int) -> bool:
        """
        :returns whether the layer with the given index has a voting weight associated with it
        """
        return str(int(layer_index)) in self.voting_weights.keys()

    def get_keys_of_active_layers(self) -> torch.Tensor:
        """
        :returns all indicies of all layers in self.layers that have a voting weight associated with them
        as ParamDict is ordered it should hold that if voting weight of layer i w_i exists 
        -> (i, w_i) in zip(get_voting_weight_keys(), get_voting_weight_values())
        :return: 1-dim tensor that contains all the indicies of all layers in self.layers with an voting weight
        """
        return torch.tensor(list(map(int, self.voting_weights.keys())), dtype=torch.int)

    def get_voting_weight_values(self) -> torch.Tensor:
        """
        :returns all voting weights of all layers in self.layers that have one associated with them
        as ParamDict is ordered it should hold that if voting weight of layer i w_i exists 
        -> (i, w_i) in zip(get_voting_weight_keys(), get_voting_weight_values())
        :return: 1-dim tensor that contains all the voting weights of all layers in self.layers
        """
        return torch.tensor(list(self.voting_weights.values()), dtype=torch.float)

    def get_weight_correction_factor(self, layer_index: int) -> float:
        """
        :returns the weight correction factor of the hidden layer at given index
        :param layer_index: the index of the hidden layer in the self.layers list
        :return: float between 0 and 1 that is used as a factor to reward/punish weights on correctly/falsely categorizing
        """
        return self.weight_correction_factor[str(layer_index)]

    def get_weight_correction_factor_keys(self) -> torch.Tensor:
        return torch.tensor(list(map(int, self.weight_correction_factor.keys())))

    def get_weight_correction_factor_values(self) -> torch.Tensor:
        return torch.tensor(list(self.weight_correction_factor.values()), dtype=torch.float)

    def get_weight_correction_factor_items(self):
        return torch.stack((self.get_weight_correction_factor_keys(), self.get_weight_correction_factor_values()))

    def __set_weight_correction_factor(self, layer_index: int, new_weight_correction_factor: float) -> None:
        self.weight_correction_factor[str(layer_index)] = new_weight_correction_factor

    def __pop_weight_correction_factor(self, layer_index: int) -> float:
        return self.weight_correction_factor.pop(str(layer_index))

    def weight_correction_factor_with_index_exists(self, layer_index: int):
        """
        :returns whether the layer with the given index has a weight correction factor associated with it
        """
        return str(layer_index) in self.weight_correction_factor.keys()
