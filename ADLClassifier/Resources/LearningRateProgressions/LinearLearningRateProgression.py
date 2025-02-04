from dataclasses import dataclass

from ADLClassifier.Resources.LearningRateProgressions.BaseLearningRateProgression import \
    BaseLearningRateProgression


@dataclass
class LinearLearningRateProgression(BaseLearningRateProgression):
    initial_learning_rate: float = 0.05
    decay_alpha: float = 0.01

    @property
    def learning_rate(self) -> float:
        return self.initial_learning_rate / (1 + self.decay_alpha * self.classifier.nr_of_instances_seen)

    @property
    def name(self) -> str:
        return f"LinearLearningRateProgression{self.decay_alpha}"
