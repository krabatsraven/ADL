from capymoa.stream.drift import AbruptDrift, RecurrentConceptDriftStream
from capymoa.stream.generator import AgrawalGenerator, SEA

from Evaluation._config import MAX_RECURRENCES_PER_STREAM, CONCEPT_LENGTH

names_for_concepts = lambda lst, dict: list(map(lambda x: dict[x], lst))
max_recurrence_for_concepts = lambda lst: MAX_RECURRENCES_PER_STREAM // len(lst)
transition_template = AbruptDrift(position=CONCEPT_LENGTH)
recurrent_drift_for_concepts = lambda lst, dic: RecurrentConceptDriftStream(
    concept_list=lst,
    max_recurrences_per_concept=max_recurrence_for_concepts(lst),
    transition_type_template=transition_template,
    concept_name_list=names_for_concepts(lst, dict),
)


agrawal_concept_names = {
    AgrawalGenerator(classification_function=1): "agrawal1",
    AgrawalGenerator(classification_function=2): "agrawal2",
    AgrawalGenerator(classification_function=3): "agrawal3",
    AgrawalGenerator(classification_function=4): "agrawal4",
}

recurrent_drift_for_agrawal_concepts = lambda lst: recurrent_drift_for_concepts(lst, agrawal_concept_names)

sea_concept_names = {
    SEA(function=1): "sea1",
    SEA(function=2): "sea2",
    SEA(function=3): "sea3",
    SEA(function=4): "sea4",
}
recurrent_drift_for_sea_concepts = lambda lst: recurrent_drift_for_concepts(lst, sea_concept_names)
