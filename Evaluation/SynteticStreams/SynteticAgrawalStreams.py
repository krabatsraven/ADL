from capymoa.stream.drift import DriftStream
from capymoa.stream.generator import AgrawalGenerator

from Evaluation.SynteticStreams._functionals import recurrent_drift_for_agrawal_concepts


agrawal_no_drift = DriftStream(
    stream=[AgrawalGenerator(classification_function=1)]
)

agrawal_single_drift_concepts = [
    AgrawalGenerator(classification_function=1),
    AgrawalGenerator(classification_function=3)
]

agrawal_single_drift = recurrent_drift_for_agrawal_concepts(agrawal_single_drift_concepts)

agrawal_three_drifts_concepts = [
    AgrawalGenerator(classification_function=1),
    AgrawalGenerator(classification_function=3),
    AgrawalGenerator(classification_function=4)
]

agrawal_three_drifts = recurrent_drift_for_agrawal_concepts(agrawal_three_drifts_concepts)

agrawal_four_drifts_concepts = [
    AgrawalGenerator(classification_function=1),
    AgrawalGenerator(classification_function=3),
    AgrawalGenerator(classification_function=4),
    AgrawalGenerator(classification_function=2)
]

agrawal_four_drifts = recurrent_drift_for_agrawal_concepts(agrawal_four_drifts_concepts)
