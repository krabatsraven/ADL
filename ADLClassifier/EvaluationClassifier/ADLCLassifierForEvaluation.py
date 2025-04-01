from collections.abc import Callable

from ADLClassifier.ExtendedClassifier import extended_classifier
from ADLClassifier.BaseClassifier import ADLClassifier
from ADLClassifier.EvaluationClassifier.EvaluationWrapper import record_emissions, record_network_graph


def extend_classifier_for_evaluation(
        *decorators: Callable[[type(ADLClassifier)], type(ADLClassifier)],
        with_emissions: bool = False
) -> type(ADLClassifier):
    """
    creates a ADLClassifier Class with the given decorators for evaluation purposes
    :param decorators: ADLClass decorators
    :param with_emissions: if true the emissions of the classifier will be tracked as well
    :return: ADLClassifier decorated with decorators and with graph network/emission tracking
    """

    if with_emissions:
        return record_network_graph(record_emissions(extended_classifier(*decorators)))
    else:
        return record_network_graph(extended_classifier(*decorators))