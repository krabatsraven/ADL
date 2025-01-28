from collections.abc import Callable

from ADLClassifier import ADLClassifier


def extended_classifier(*decorators: Callable[[type(ADLClassifier)], type(ADLClassifier)]) -> type(ADLClassifier):
    """
    creates an ADLClassifier that was decorated with @decorators
    :param decorators: number of functions to decorate the BaseADLClassifier with
    :return: 
    """
    extended_classifier = ADLClassifier
    for decorator in decorators:
        extended_classifier = decorator(extended_classifier)

    return extended_classifier
