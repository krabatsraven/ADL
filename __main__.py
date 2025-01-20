from pathlib import Path

import torch
from capymoa.datasets import Electricity, ElectricityTiny

from ADLClassifier import ADLClassifierWithGraphRecord, WinningLayerADLClassifierWithGraphRecord
from Evaluation import __get_run_id, __evaluate_on_stream, __plot_and_save_result, __compare_all_of_one_run, \
    __compare_results_via_plot_and_save
from Evaluation.EvaluationFunctions import _evaluate_parameters

def _test_example(run: bool):

    if run:
        streams = [
            # ElectricityTiny(),
            Electricity()
        ]
        learning_rates = [5e-2, 1e-2, 1e-3, 1e-4]
        mci_thresholds = [1e3, 1e-10, 1e-20, 1e-50]
        classifiers = [WinningLayerADLClassifierWithGraphRecord, ADLClassifierWithGraphRecord]

        _evaluate_parameters(
            adl_classifiers=classifiers,
            streams=streams,
            learning_rates=learning_rates,
            mci_thresholds=mci_thresholds
        )


if __name__ == "__main__":
    _test_example(True)
