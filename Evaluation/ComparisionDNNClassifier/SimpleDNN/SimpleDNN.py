from typing import List

import torch
from torch import nn


class SimpleDNN(nn.Module):
    def __init__(
            self,
            nr_of_features: int,
            nr_of_classes: int,
            nr_of_nodes_in_layers: List[int],
            *args,
            **kwargs
    ) -> None:
        """
        simple dnn of linear layers with sigmoid activation functions and a softmax activation function for the output layer,
        the layers are constructed with the number of nodes (output size) specified by the user,
        the final prediction is done by argmax of the softmax
        :param nr_of_features: nr of features in the input data
        :param nr_of_classes: nr of classes the network should predict
        :param nr_of_nodes_in_layers: a list of the number of nodes the hidden layers should have
        """
        super().__init__(*args, **kwargs)

        assert len(nr_of_nodes_in_layers) > 0, "at least one hidden layer needs to be spezified for a dnn"

        self.nr_of_features = nr_of_features
        self.nr_of_classes = nr_of_classes
        self.model_structure = nr_of_nodes_in_layers

        list_of_all_sizes = [nr_of_features, *nr_of_nodes_in_layers, nr_of_classes]
        layers = [
            layer
            for nr_of_nodes_in_previous, nr_of_nodes_in_current in
            zip(list_of_all_sizes[:-1], list_of_all_sizes[1:])
            for layer in (nn.Linear(nr_of_nodes_in_previous, nr_of_nodes_in_current), nn.Sigmoid())
        ]
        # remove the sigmoid of the output layer
        layers = layers[:-1]
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)
        self.net.apply(self._initialise_weights)

    def _initialise_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns the probabilities for the classes from self.nr_of_classes many classes stemming from the data input x
        alternative: defines the forward pass through the network
        :param x: batch of data to be classified with self.input_size many features
        :return: probability tensor that contains the probabilities of the classes
        """
        return self.net(x)

    def print(self) -> None:
        print(self.net)
