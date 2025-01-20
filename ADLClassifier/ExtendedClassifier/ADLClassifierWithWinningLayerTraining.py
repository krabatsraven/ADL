from ADLClassifier.BaseClassifier.ADLClassifier import ADLClassifier
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import train_only_winning_layer


@train_only_winning_layer
class ADLClassifierWithWinningLayerTraining(ADLClassifier):
    pass
