from ADLClassifier.BaseClassifier.ADLClassifier import ADLClassifier
from ADLClassifier.EvaluationClassifier.EvaluationWrapper import record_emissions, record_network_graph


@record_network_graph
class ADLClassifierWithGraphRecord(ADLClassifier):
    pass


@record_emissions
@record_network_graph
class ADLClassifierWithGraphRecordAndEmissionTracking(ADLClassifier):
    pass
