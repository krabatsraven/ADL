from pathlib import Path

from capymoa.datasets import Electricity, ElectricityTiny

from ADLClassifier import ADLClassifierWithGraphRecord, WinningLayerADLClassifierWithGraphRecord
from Evaluation import __get_run_id, __evaluate_on_stream, __plot_and_save_result, __compare_all_of_one_run, \
    __compare_results_via_plot_and_save

if __name__ == "__main__":

    run = False

    if run:
        streams = [
            ElectricityTiny(),
            # Electricity()
        ]
        learning_rates = [5e-2, 1e-2, 1e-3, 1e-4]
        mci_thresholds = [1e3, 1e-10, 1e-20, 1e-50]
        classifiers = [WinningLayerADLClassifierWithGraphRecord, ADLClassifierWithGraphRecord]

        run_id = __get_run_id()

        for classifier in classifiers:
            for stream_data in streams:
                for lr in learning_rates:
                    for mci_threshold in mci_thresholds:
                        __evaluate_on_stream(
                            stream_data=stream_data, 
                            learning_rate=lr, 
                            threshold_for_layer_pruning=mci_threshold, 
                            run_id=run_id,
                            classifier=classifier
                        )
            __plot_and_save_result(run_id, show=False)
            __compare_all_of_one_run(run_id, show=True)

    # __plot_and_save_result(9, show=False)
    # __plot_and_save_result(31, show=False)

    run_7_7_8_and_9 = [
        Path("/home/david/PycharmProjects/ADL/results/runs/runID=7/lr=0.001_MCICutOff=1e-07/electricity_tiny"),
        Path("/home/david/PycharmProjects/ADL/results/runs/runID=7/lr=0.001_MCICutOff=1e-07/electricity_tiny"),
        Path("/home/david/PycharmProjects/ADL/results/runs/runID=8/lr=0.001_MCICutOff=1e-07/electricity_tiny"),
        Path("/home/david/PycharmProjects/ADL/results/runs/runID=9/lr=0.001_MCICutOff=1e-07_classifier=ADLClassifier/electricity_tiny")
    ]
    __compare_all_of_one_run(30, show=False)
    __compare_all_of_one_run(33, show=False)
    __compare_results_via_plot_and_save(run_7_7_8_and_9, show=False)