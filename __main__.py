import numpy as np
import pandas as pd
from capymoa.datasets import ElectricityTiny

from ADLClassifier.EvaluationClassifier import *
from ADLClassifier.Resources import *
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import *
from ADLClassifier.Resources.LearningRateProgressions.ExponentialLearningRateProgression import \
    ExponentialLearningRateProgression

from Evaluation.EvaluationFunctions import _evaluate_parameters


def _test_example(run: bool):

    if run:
        streams = [
            ElectricityTiny(),
            # Electricity()
        ]
        learning_rates = [
            # 5e-1,
            # 1e-1,
            # 5e-2,
            # 1e-2,
            # 1e-3,
            # 1e-4,
            LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
            LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
            ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
            ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
            # exponential_decay(1)(0.01)
            ]
        # anything above -15 is above the precision of python float
        mci_thresholds = [
            1e-5, 1e-6,
            1e-7,
            1e-8, 1e-9,
            1e-10, 1e-11, 1e-12, 1e-13, 1e-14
        ]
        classifiers = [
            extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop),
            # extend_classifier_for_evaluation(vectorized_for_loop),
            # extend_classifier_for_evaluation(winning_layer_training),
        ]

        adwin_deltas=[
            1e-1, 1e-2, 1e-3, 1e-4,
                      1e-5,
            1e-6, 1e-7, 1e-8, 1e-9, 1e-10
            ]
        grace_periods_global = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        grace_periods_for_layer = grace_periods_global

        _evaluate_parameters(
            adl_classifiers=classifiers,
            streams=streams,
            learning_rates=learning_rates,
            # mci_thresholds=mci_thresholds,
            # adwin_deltas=adwin_deltas,
            # grace_periods_global=grace_periods_global,
            # grace_periods_for_layer=grace_periods_for_layer,
        )


if __name__ == "__main__":
    _test_example(True)
    # __plot_and_save_result(49)
