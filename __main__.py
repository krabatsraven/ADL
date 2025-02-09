import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from capymoa.classifier import OnlineBagging
from capymoa.datasets import ElectricityTiny, Electricity, CovtypeTiny
from capymoa.evaluation import prequential_evaluation
from capymoa.evaluation.visualization import plot_windowed_results
from capymoa.stream import Stream
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift
from capymoa.stream.generator import SEA
from moa.streams.generators import AgrawalGenerator

from ADLClassifier.EvaluationClassifier import *
from ADLClassifier.Resources import *
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import *
from Evaluation import simple_agraval_single_drift, simple_agraval_three_drifts
from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier

from Evaluation.EvaluationFunctions import _evaluate_parameters, __write_summary, __get_run_id


def _test_example(name: Optional[str] = None):

    streams = [
        # ElectricityTiny(),
        CovtypeTiny(),
        # Electricity()
    ]
    learning_rates = [
        # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.1),
        # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
        # LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
        # LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.1),
        # LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.01),
        # LinearLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.001),

        # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.1),
        # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.01),
        # ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001),
        # ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.1),
        # ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.01),
        # ExponentialLearningRateProgression(initial_learning_rate=0.5, decay_alpha=0.001),
        5e-1,
        # 1e-1,
        5e-2,
        # 1e-2,
        1e-3
        ]

    mci_thresholds = [
        # 1e-5,
        1e-6, 1e-7, 1e-8
    ]
    classifiers = [
        extend_classifier_for_evaluation(winning_layer_training, vectorized_for_loop),
        # extend_classifier_for_evaluation(winning_layer_training),
    ]

    adwin_deltas=[
        # 1e-1, 1e-2,
        1e-3, 
        # 1e-4,
        1e-5, 
        # 1e-6,
        1e-7,
        # 1e-8, 1e-9, 1e-10
    ]

    grace_periods_for_layer = [
        4, 8, 16, 32, None
    ]
    grace_periods_global = None

    run_id = __get_run_id()

    _evaluate_parameters(
        adl_classifiers=classifiers,
        streams=streams,
        learning_rates=learning_rates,
        mci_thresholds=mci_thresholds,
        adwin_deltas=adwin_deltas,
        grace_periods_global=grace_periods_global,
        grace_periods_for_layer=grace_periods_for_layer,
    )

    if name is not None:
        folder = Path("/home/david/PycharmProjects/ADL/results/experiment_data_selected") / name
        run_folder = Path(f"/home/david/PycharmProjects/ADL/results/runs/runID={run_id}")
        comparision_folder = Path("/home/david/PycharmProjects/ADL/results/comparisons/comparison=0")

        shutil.move(run_folder, folder)
        shutil.move(comparision_folder, folder)


if __name__ == "__main__":
    # _test_example(True)
    stream = Electricity()
    learner = SimpleDNNClassifier(
        schema=stream.get_schema(),
        model_structure=[2**10, 2**11],
        lr=0.001
    )

    results_sea2drift_OB = prequential_evaluation(
        stream=stream, learner=learner, window_size=1, max_instances=40000
    )

    print(
        f"The definition of the DriftStream is accessible through the object:\n {stream}"
    )
    plot_windowed_results(results_sea2drift_OB, metric="accuracy")
    print(f"accuracy={results_sea2drift_OB.cumulative.metrics_dict()['accuracy']}")

