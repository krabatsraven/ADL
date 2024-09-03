from typing import List, Dict

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

        # assumes that at least a single concept is to be learned in an data stream
        # list of all dynamic layers, starting with a single layer in the shape (in, 1)
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(nr_of_features, 1)])

        # list of all linear layers for the voting part, starts with a single layer in the shape (1, out)
        # contains only the indices of self.layers that are eligible to vote
        self.voting_linear_layers: nn.ModuleDict = nn.ModuleDict({'0': nn.Linear(1, nr_of_classes)})

        # all voting weights should always be normalised,
        # and only contain the indices of self.layers that eligible to vote
        self.voting_weights: Dict[int, float] = {0: 1.0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # section 4.1 equation 1:

        # calculate all h^{l} = \sigma(W^{l} h^{(l-1)} + b^{l}), h^{(0)} = x
        hidden_layers: List[torch.Tensor] = [x := nn.Sigmoid()(layer(x)) for layer in self.layers]

        # calculate all y^i = s.max(W_{s_l}h^{l} + b_{s_l})
        # todo: check that below is correct
        # that are not currently pruned
        voted_class_probabilities = [torch.mul(nn.Softmax()(self.voting_linear_layers[str(i)](hidden_layers[i])), beta)
                                     for i, beta in self.voting_weights.items()]

        # calculated voted/weighted class probability
        total_weighted_class_probability = torch.stack(voted_class_probabilities, dim=0).sum(dim=0)

        # classification by majority rule
        # todo: according to paper max, but shouldn't it be argmax?
        return torch.max(total_weighted_class_probability)

    def _add_layer(self) -> None:
        """
        adds a new untrained layer
        """

        previous_layer_output_size, previous_layer_input_size = self.layers[-1].weight.size()
        
        # new layers are initialised with one node
        # todo: that means out=1?
        nr_of_out_nodes = 1
        new_layer = nn.Linear(previous_layer_output_size, nr_of_out_nodes)
        self.layers.append(new_layer)

        idx_of_new_layer = len(self.layers) - 1
        self.voting_linear_layers[str(idx_of_new_layer)] = nn.Linear(nr_of_out_nodes, self.output_size)
        self.voting_weights[idx_of_new_layer] = 0

    def _add_node(self, layer_index: int) -> None:
        """
        adds a new node to the layer l_{layer_index}

        :param layer_index: the index of the layer in the list of layers: self.layers
        """
        # todo: find layer
        # todo: generate layer with the same weights and one additional one
        # todo: the new weight is set by Xavier Initialisation
        # todo: change the shape Ws^l
        # (the linear layer that takes the outputs of the hidden layer for the voting function)
        raise NotImplementedError
