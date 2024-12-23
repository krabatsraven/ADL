import time

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

    print(f"the learner has {len(adl_classifier.model.layers)} hidden layers and {len(adl_classifier.model.voting_linear_layers)} output layers")
    print(f"the active hidden layers of the model have {dict((adl_classifier.model.transform_layer_index_to_output_layer_key(i), tuple(adl_classifier.model.layers[i].weight.size())) for i in adl_classifier.model.active_layer_keys())} shaped weight matricies")
