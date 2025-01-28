from ADLClassifier import ADLClassifier
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import vectorized_for_loop, winning_layer_training, global_grace_period, grace_period_per_layer


@winning_layer_training
class WinningLayerADLClassifier(ADLClassifier):
    pass


@vectorized_for_loop
class ADLClassifierWithoutForLoop(ADLClassifier):
    pass


@winning_layer_training
@vectorized_for_loop
class WinningLayerADLCLassifierWithoutForLoop(ADLClassifier):
    pass


@global_grace_period
class ADLClassifierWithGlobalGracePeriod(ADLClassifier):
    pass


@grace_period_per_layer
class ADLClassifierWithGracePeriodPerLayer(ADLClassifier):
    pass


@winning_layer_training
@global_grace_period
class WinningLayerADLClassifierWithGlobalGracePeriod(ADLClassifier):
    pass


@winning_layer_training
@grace_period_per_layer
class WinningLayerADLClassifierWithGracePeriodPerLayer(ADLClassifier):
    pass


@vectorized_for_loop
@global_grace_period
class ADLClassifierWithGlobalGracePeriodWithoutForLoop(ADLClassifier):
    pass


@vectorized_for_loop
@grace_period_per_layer
class ADLClassifierWithGracePeriodPerLayerWithoutForLoop(ADLClassifier):
    pass


@vectorized_for_loop
@winning_layer_training
@global_grace_period
class WinningLayerADLClassifierWithGlobalGracePeriodWithoutForLoop(ADLClassifier):
    pass


@vectorized_for_loop
@winning_layer_training
@grace_period_per_layer
class WinningLayerADLClassifierWithGracePeriodPerLayerWithoutForLoop(ADLClassifier):
    pass
