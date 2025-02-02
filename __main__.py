from capymoa.datasets import Electricity, ElectricityTiny
from capymoa.drift.detectors import ADWIN

from ADLClassifier.EvaluationClassifier import *
from ADLClassifier.ExtendedClassifier.FunctionalityWrapper import *
from Evaluation import __plot_and_save_result

from Evaluation.EvaluationFunctions import _evaluate_parameters, __write_summary, ADWIN_DELTA_STANDIN


def _test_example(run: bool):

    if run:
        streams = [
            ElectricityTiny(),
            # Electricity()
        ]
        learning_rates = [
            5e-2,
            # 1e-2,
            # 1e-3,
            # 1e-4
            ]
        # anything above -15 is above the precision of python float
        mci_thresholds = [
            # 1e3, 5e-2, 1e-2, 1e-3, 1e-4,
            1e-5, # 1e-6, 1e-7,
            # 1e-8, 1e-9,
            # 1e-10, 1e-11, 1e-12, 1e-13, 1e-14,
            # 1e-15, 1e-50
        ]
        classifiers = [
            extend_classifier_for_evaluation(grace_period_per_layer(10), winning_layer_training, vectorized_for_loop),
        ]

        _evaluate_parameters(
            adl_classifiers=classifiers,
            streams=streams,
            learning_rates=learning_rates,
            mci_thresholds=mci_thresholds,
            user_added_parameters={
                ADWIN_DELTA_STANDIN: [1e-5]
            }
        )


if __name__ == "__main__":
    _test_example(True)
    # __write_summary(49, {"delta"})
    # __plot_and_save_result(49)
