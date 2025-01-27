from ADLClassifier import ADLClassifier
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import eliminate_for_loop, train_only_winning_layer


@eliminate_for_loop
class ADLClassifierWithoutForLoop(ADLClassifier):
    pass

@train_only_winning_layer
@eliminate_for_loop
class WinningLayerADLCLassifierWithoutForLoop(ADLClassifier):
    pass
