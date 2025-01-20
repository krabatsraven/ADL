import os
import time
from pathlib import Path

import pandas as pd
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import Stream

from ADLClassifier import ADLClassifier

def __get_run_id() -> int:
    results_dir_path = Path("results/runs/")
    results_dir_path.mkdir(parents=True, exist_ok=True)
    if any(results_dir_path.iterdir()):
        *_, elem = results_dir_path.iterdir()
        name, running_nr = elem.stem.split("=")
        return int(running_nr) + 1
    else:
        return 1


def __evaluate_on_stream(
        stream_data: Stream,
        learning_rate: float,
        threshold_for_layer_pruning: float,
        run_id: int,
        classifier: type(ADLClassifier)
) -> None:


    adl_classifier = classifier(
        schema=stream_data.schema,
        lr=learning_rate,
        mci_threshold_for_layer_pruning=threshold_for_layer_pruning
    )

    assert hasattr(adl_classifier, "record_of_model_shape"), f"ADL classifier {adl_classifier} does not keep track of model shape, and cannot be evaluated"

    name_string_of_stream_data = f"{stream_data._filename.split('.')[0]}/"
    hyperparameter_part_of_name_string = f"lr={learning_rate}_MCICutOff={threshold_for_layer_pruning}_classifier={adl_classifier.__str__()}"
    results_dir_path = Path("results/runs")
    results_dir_path.mkdir(parents=True, exist_ok=True)

    results_path = results_dir_path / f"runID={run_id}" / hyperparameter_part_of_name_string / name_string_of_stream_data
    results_path.mkdir(parents=True, exist_ok=True)

    os.environ["CODECARBON_OUTPUT_DIR"] = str(results_path)
    os.environ["CODECARBON_TRACKING_MODE"] = "process"
    os.environ["CODECARBON_LOG_LEVEL"] = "CRITICAL"

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier, window_size=1, optimise=True, store_predictions=False, store_y=False)
    total_time_end = time.time_ns()

    # todo: delete print statement and time to train as it measures part of the evaluating process and will become obsolete shortly:
    print(stream_data._filename)
    print(f"total time spend in covariance loop: {adl_classifier.total_time_in_loop:.2E}ns, that equals {adl_classifier.total_time_in_loop / 10 ** 9:.2f}s or {adl_classifier.total_time_in_loop / 10 ** 9 /60:.2}min")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"meaning that the covariance loop alone took {adl_classifier.total_time_in_loop / (total_time_end - total_time_start) * 100}% of the training time")

    print(f"\n\tAll the cumulative results:")
    print(results_ht.cumulative.metrics_dict())

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    # add custom parameters to result dict
    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end.insert(loc=0, column=key, value=[adl_classifier.record_of_model_shape[key][-1]])
        # metrics_at_end[key] = adl_classifier.record_of_model_shape[key][-1]
        windowed_results[key] = adl_classifier.record_of_model_shape[key]

    metrics_at_end.to_pickle(results_path / "metrics.pickle")
    windowed_results.to_pickle(results_path / "metrics_per_window.pickle")

    results_ht.write_to_file(results_path.absolute().as_posix())