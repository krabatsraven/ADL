from dataclasses import dataclass

import numpy as np

from ADLClassifier.Resources.LearningRateProgressions import BaseLearningRateProgression


@dataclass
class ExponentialLearningRateProgression(BaseLearningRateProgression):
    initial_learning_rate: float = 0.05
    decay_alpha: float = 0.01

    @property
    def learning_rate(self) -> float:
        return self.initial_learning_rate * np.exp(-self.decay_alpha * self.classifier.nr_of_instances_seen)

    @property
    def name(self) -> str:
        return f"ExponentialLearningRateProgression({self.initial_learning_rate})({self.decay_alpha})"
