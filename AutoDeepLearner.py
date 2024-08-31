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

        # list of all dynamic layers, starting with a single layer in the shape (in, out)
        self.layers: nn.ModuleList = nn.ModuleList([nn.Linear(nr_of_features, 1)])
        self.voting_linear_layers: nn.ModuleDict = nn.ModuleDict({'0': nn.Linear(1, nr_of_classes)})

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
        # todo: check that dims are correct/ that sum is along correct axis
        total_weighted_class_probability = torch.stack(voted_class_probabilities, dim=0).sum(dim=0)

        # classification by majority rule
        # todo: according to paper max, but shouldn't it be argmax?
        return torch.max(total_weighted_class_probability)

    def __add_layer(self) -> None:
        # todo: this is wip, might all be stupid, needs adding
        previous_layer_output_size = self.layers[-1].weights.size()[0]

        # new layers are initialised with one node
        # todo: that means out=1?
        new_layer = nn.Linear(previous_layer_output_size, 1)
        self.layers.append(new_layer)
        self.voting_linear_layers[str(len(self.layers) - 1)] = nn.Linear(1, self.output_size)

        # todo: train new layer

        # todo: change voting weights

        raise NotImplementedError

    def __add_node(self) -> None:
        # todo: change Ws^l
        raise NotImplementedError
