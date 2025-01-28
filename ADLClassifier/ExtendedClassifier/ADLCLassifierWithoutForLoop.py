from ADLClassifier import ADLClassifier
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import vectorized_for_loop, winning_layer_training


@vectorized_for_loop
class ADLClassifierWithoutForLoop(ADLClassifier):
    pass

@winning_layer_training
@vectorized_for_loop
class WinningLayerADLCLassifierWithoutForLoop(ADLClassifier):
    pass
