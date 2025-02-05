from pathlib import Path

import numpy as np
import pandas as pd
from capymoa.datasets import ElectricityTiny

from ADLClassifier.EvaluationClassifier import *
from ADLClassifier.Resources import *
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import *
from ADLClassifier.Resources.LearningRateProgressions.ExponentialLearningRateProgression import \
    ExponentialLearningRateProgression
from Evaluation import __compare_results_via_plot_and_save, __plot_and_save_result

from Evaluation.EvaluationFunctions import _evaluate_parameters, __write_summary
from Evaluation.EvaluationFunctions import __write_summary


def _test_example(run: bool):

    if run:
        streams = [
            ElectricityTiny(),
            # Electricity()
        ]
        learning_rates = [
            LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
            # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
            # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
            # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
            # 5e-1,
            # 1e-1,
            # 5e-2,
            # 1e-2,
            # 1e-3,
            # 1e-4
            ]
        # anything above -15 is above the precision of python float
        mci_thresholds = [
            1e-5, 1e-6,
            1e-7,
            1e-8, 1e-9,
            1e-10, 1e-11, 1e-12, 1e-13, 1e-14
        ]
        # todo: check effect of grace period, learning rate progression
        classifiers = [
            # extend_classifier_for_evaluation(vectorized_for_loop),
            extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop),
            # extend_classifier_for_evaluation(winning_layer_training),
        ]

        adwin_deltas=[
            # 1e-1, 1e-2, 1e-3, 1e-4,
                      1e-5,
            # 1e-6, 1e-7, 1e-8, 1e-9, 1e-10
            ]
        grace_periods_global = [
            # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1024
        ]
        grace_periods_for_layer = grace_periods_global

        _evaluate_parameters(
            adl_classifiers=classifiers,
            streams=streams,
            learning_rates=learning_rates,
            # mci_thresholds=mci_thresholds,
            adwin_deltas=adwin_deltas,
            grace_periods_global=grace_periods_global,
            grace_periods_for_layer=grace_periods_for_layer,
        )


if __name__ == "__main__":
    # __write_summary(2, {"lr", "MCICutOff", "classifier", "adwin-delta", "globalGracePeriod", "gracePeriodPerLayer"})
    _test_example(True)
    # __plot_and_save_result(2)
    list_of_interest_high_acc = [
        Path("results/runs/runID=2/classifier=ADLClassifierWithoutForLoopWithGraphRecord_adwin-delta=1e-10_lr=0.5_MCICutOff=1e-05_gracePeriodPerLayer=2/electricity_tiny"),
        Path("results/runs/runID=2/classifier=ADLClassifierWithoutForLoopWithGraphRecord_adwin-delta=1e-06_lr=0.5_MCICutOff=1e-05_gracePeriodPerLayer=2/electricity_tiny"),
        Path("results/runs/runID=2/classifier=ADLClassifierWithoutForLoopWithGraphRecord_adwin-delta=0.1_lr=0.5_MCICutOff=1e-05_gracePeriodPerLayer=4/electricity_tiny"),
        Path("results/runs/runID=2/classifier=ADLClassifierWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.5_MCICutOff=1e-05_gracePeriodPerLayer=2/electricity_tiny"),
        Path("results/runs/runID=2/classifier=ADLClassifierWithoutForLoopWithGraphRecord_adwin-delta=0.01_lr=0.5_MCICutOff=1e-05_globalGracePeriod=2/electricity_tiny")
    ]
    # __compare_results_via_plot_and_save(list_of_interest_high_acc, show=False)

    list_of_different_lrs = [
        Path("results/experiment_data_selected/different_lrs/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.5_MCICutOff=1e-07/electricity_tiny"),
        Path("results/experiment_data_selected/different_lrs/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.1_MCICutOff=1e-07/electricity_tiny"),
        Path("results/experiment_data_selected/different_lrs/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07/electricity_tiny"),
        Path("results/experiment_data_selected/different_lrs/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.01_MCICutOff=1e-07/electricity_tiny")
    ]
    # __compare_results_via_plot_and_save(list_of_different_lrs, show=False)
    
    list_of_different_layer_graces = [
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=4/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=8/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=16/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=32/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=64/electricity_tiny")
    ]
    # __compare_results_via_plot_and_save(list_of_different_layer_graces, show=False)
    
    list_of_different_global_graces = [
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=4/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=8/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=16/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=32/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=64/electricity_tiny")
    ]
    # __compare_results_via_plot_and_save(list_of_different_global_graces, show=False)
    
    list_of_global_and_layer_graces = [
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=4/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=2/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=2/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=8/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_gracePeriodPerLayer=8/electricity_tiny"),
        Path("results/experiment_data_selected/different_grace_periods/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07_globalGracePeriod=4/electricity_tiny"),
        Path("results/experiment_data_selected/different_mcis/classifier=ADLClassifierWithWinningLayerTrainingWithoutForLoopWithGraphRecord_adwin-delta=1e-05_lr=0.05_MCICutOff=1e-07/electricity_tiny")
    ]
    # __compare_results_via_plot_and_save(list_of_global_and_layer_graces, show=False)
