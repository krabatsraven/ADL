from typing import Optional, List

import numpy as np
import torch
from capymoa.base import Classifier
from capymoa.instance import Instance, LabeledInstance
from capymoa.stream import Schema
from capymoa.type_alias import LabelProbabilities, LabelIndex
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.optim import Optimizer, SGD

from ADLClassifier.ExtendedClassifier.FunctionalityWrapper.input_preprocessing import StreamingStandardScaler
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
        self.schema = schema

        transformers = [
            elem
            for elem in
            [
                (OneHotEncoder(categories=[torch.arange(len(self.schema.get_moa_header().attribute(i).getAttributeValues()), dtype=torch.float32)]), [i])
                if self.schema.get_moa_header().attribute(i).isNominal()
                else (StreamingStandardScaler(), [i])
                if self.schema.get_moa_header().attribute(i).isNumeric()
                else None
                for i in range(self.schema.get_num_attributes())
            ]
            if elem is not None
        ]
        self.input_transformer = make_column_transformer(
            *transformers,
            remainder='passthrough',
            sparse_threshold=0)

        self.nr_of_instances_seen = 0
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

    def _preprocess_instance(self, instance):
        if len(instance.x.shape) < 2:
            # adding a dimension to data to ensure shape of [batch_size(=1), nr_of_features]
            X = instance.x[np.newaxis, ...]
        else:
            X = instance.x

        if self.nr_of_instances_seen == 0:
            self.input_transformer.fit(X)
            self.nr_of_instances_seen += 1

        else:
            self.nr_of_instances_seen += 1
        return torch.from_numpy(self.input_transformer.transform(X)).to(device=self.device, dtype=torch.float)

    @property
    def state_dict(self):
        return {
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.learning_rate,
            'device': self.device,
            'nr_of_instances_seen': self.nr_of_instances_seen,
        }

    @state_dict.setter
    def state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.learning_rate = state_dict['lr']
        self.device = state_dict['device'],
        self.nr_of_instances_seen = state_dict['nr_of_instances_seen']
