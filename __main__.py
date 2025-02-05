import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from capymoa.datasets import ElectricityTiny

from ADLClassifier.EvaluationClassifier import *
from ADLClassifier.Resources import *
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import *
from ADLClassifier.Resources.LearningRateProgressions.ExponentialLearningRateProgression import \
    ExponentialLearningRateProgression
from Evaluation import __compare_results_via_plot_and_save, __plot_and_save_result, __compare_all_of_one_run

from Evaluation.EvaluationFunctions import _evaluate_parameters, __write_summary
from Evaluation.EvaluationFunctions import __write_summary


def _test_example(run: bool):

    if run:
        streams = [
            ElectricityTiny(),
        ]
        learning_rates = [
            LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.1),
            LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
            LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
            LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.1),
            LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.01),
            LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.001),

            ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.1),
            ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
            ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
            ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.1),
            ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.01),
            ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.001),
            5e-1,
            1e-1,
            5e-2,
            1e-2,
            1e-3
            ]

        mci_thresholds = [
            1e-5, 1e-6, 1e-7, 1e-8
        ]
        classifiers = [
            extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop),
        ]

        adwin_deltas=[
            1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7
        ]

        grace_periods_global = [
            4, 8, 16, 32, None
        ]
        grace_periods_for_layer = grace_periods_global

        _evaluate_parameters(
            adl_classifiers=classifiers,
            streams=streams,
            learning_rates=learning_rates,
            mci_thresholds=mci_thresholds,
            adwin_deltas=adwin_deltas,
            grace_periods_global=grace_periods_global,
            grace_periods_for_layer=grace_periods_for_layer,
        )


if __name__ == "__main__":
    _test_example(True)
    folder = Path("/home/david/PycharmProjects/ADL/results/experiment_data_selected") /"grid_on_tiny"
    run_folder = Path("/home/david/PycharmProjects/ADL/results/runs/runID=1")
    comparision_folder = Path("/home/david/PycharmProjects/ADL/results/comparisons/comparison=0")

    shutil.move(run_folder, folder)
    shutil.move(comparision_folder, folder)
