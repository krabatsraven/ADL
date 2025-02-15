from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from ADLClassifier.BaseClassifier import ADLClassifier


@dataclass
class BaseLearningRateProgression:
    __classifier: Optional[ADLClassifier] = None

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def classifier(self) -> ADLClassifier:
        if self.__classifier is None:
            raise ValueError("No classifier")
        else:
            return self.__classifier

    @classifier.setter
    def classifier(self, classifier: ADLClassifier) -> None:
        assert isinstance(classifier, ADLClassifier), 'classifier must be a ADLClassifier'
        self.__classifier = classifier

    def __call__(self) -> float:
        return self.learning_rate

    def __str__(self):
        return self.name