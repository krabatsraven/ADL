from typing import Optional, List

import numpy as np
import torch
from capymoa.base import Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.stream import Schema
from capymoa.type_alias import LabelProbabilities, LabelIndex
from torch import nn
from torch.optim import Optimizer, SGD

from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNN import SimpleDNN


class SimpleDNNClassifier(Classifier):
    def __init__(
            self, 
            schema: Schema,
            model_structure: List[int],
            random_seed: int = 1,
            type_of_optimizer: str = "SGD",
            loss_fn=lambda predicted_props, truth: nn.NLLLoss()(torch.log(predicted_props), truth),
            device: str = "cpu",
            lr: float = 1e-3
    ):
        super().__init__(schema, random_seed)
        self.learning_rate = lr
        self.device = device
        self.loss_fn = loss_fn

        self.model = SimpleDNN(
            nr_of_features=schema.get_num_attributes(),
            nr_of_classes=schema.get_num_classes(),
            nr_of_nodes_in_layers=model_structure,
        ).to(self.device)

        self._initialize_optimizer(type_of_optimizer, lr)

    def _initialize_optimizer(self, type_of_optimizer: str, lr: float) -> None:
        match type_of_optimizer:
            case "SGD":
                self.optimizer = SGD(lr=lr, params=self.model.parameters())
            case _:
                raise ValueError("Unknown optimizer type")

    def __str__(self):
        return f"SimpleDNNClassifier({', '.join(map(str, self.model.model_structure))})"

    def train(self, instance: LabeledInstance):
        X = torch.tensor(instance.x, dtype=torch.float32)
        y = torch.tensor(instance.y_index, dtype=torch.long)
        X, y = torch.unsqueeze(X.to(self.device), 0), torch.unsqueeze(y.to(self.device),0)
        self.optimizer.zero_grad()
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()

    def predict(self, instance: Instance):
        return self.predict_proba(instance).argmax()

    def predict_proba(self, instance: Instance):
        X = torch.unsqueeze(torch.tensor(instance.x, dtype=torch.float32).to(self.device), 0)
        with torch.no_grad():
            pred = self.model(X).detach().cpu().numpy()
        return pred
