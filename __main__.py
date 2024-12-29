import time
from pathlib import Path

import pandas as pd
from capymoa.evaluation.visualization import plot_windowed_results


from ADLClassifier import ADLClassifier

from capymoa.datasets import Electricity, ElectricityTiny
from capymoa.evaluation import prequential_evaluation

if __name__ == "__main__":

    ### capymoa training of classifier on a stream:
    ### -------------------------------------------
    elec_stream = ElectricityTiny()
    adl_classifier = ADLClassifier(schema=elec_stream.schema)

    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=elec_stream, learner=adl_classifier, window_size=100, optimise=True, store_predictions=False, store_y=False)
    total_time_end = time.time_ns()

    print(f"total time spend in covariance loop: {adl_classifier.total_time_in_loop:.2E}ns, that equals {adl_classifier.total_time_in_loop / 10 ** 9:.2f}s or {adl_classifier.total_time_in_loop / 10 ** 9 /60:.2}min")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"meaning that the covariance loop alone took {adl_classifier.total_time_in_loop / (total_time_end - total_time_start) * 100}% of the training time")

    print("\tDifferent ways of accessing metrics:")

    print(f"results_ht['wallclock']: {results_ht['wallclock']} results_ht.wallclock(): {results_ht.wallclock()}")
    print(f"results_ht['cpu_time']: {results_ht['cpu_time']} results_ht.cpu_time(): {results_ht.cpu_time()}")

    print(f"results_ht.cumulative.accuracy() = {results_ht.cumulative.accuracy()}")
    print(f"results_ht.cumulative['accuracy'] = {results_ht.cumulative['accuracy']}")
    print(f"results_ht['cumulative'].accuracy() = {results_ht['cumulative'].accuracy()}")
    print(f"results_ht.accuracy() = {results_ht.accuracy()}")

    print(f"\n\tAll the cumulative results:")
    print(results_ht.cumulative.metrics_dict())

    print(f"\n\tAll the windowed results:")

    plot_windowed_results(results_ht, metric= "accuracy")

    results_dir_path = Path("results") / f"{elec_stream._filename.split('.')[0]}/"
    results_dir_path.mkdir(parents=True, exist_ok=True)
    try:
        *_, elem = results_dir_path.iterdir()
        name, running_nr = elem.stem.split("_")
        results_path = results_dir_path / (name + "_" + str(int(running_nr) + 1))
    except ValueError:
        results_path = results_dir_path / "run_1"
    results_path.mkdir(parents=True, exist_ok=True)

    metrics_at_end = pd.DataFrame([adl_classifier.evaluator.metrics()], columns=adl_classifier.evaluator.metrics_header())

    windowed_results = adl_classifier.evaluator.metrics_per_window()

    for key in adl_classifier.record_of_model_shape.keys():
        metrics_at_end[key] = str(adl_classifier.record_of_model_shape[key][-1])
        windowed_results[key] = adl_classifier.record_of_model_shape[key]

    metrics_at_end.to_csv(results_path / "metrics.csv")
    windowed_results.to_csv(results_path / "metrics_per_window.csv")

    results_ht.write_to_file(results_path.absolute().as_posix())
    print(f"the learner has {len(adl_classifier.model.layers)} hidden layers and {len(adl_classifier.model.voting_linear_layers)} output layers")
    print(f"the active hidden layers of the model have {dict((adl_classifier.model.transform_layer_index_to_output_layer_key(i), tuple(adl_classifier.model.layers[i].weight.size())) for i in adl_classifier.model.active_layer_keys())} shaped weight matricies")