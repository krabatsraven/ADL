from capymoa.stream.drift import DriftStream
from capymoa.stream.generator import SEA

from Evaluation.SynteticStreams._functionals import recurrent_drift_for_sea_concepts

sea_no_drift = DriftStream(
    stream=[
        SEA(function=1),
    ]
)

sea_single_drift_concepts = [
    SEA(function=1),
    SEA(function=3),
]

sea_single_drift = recurrent_drift_for_sea_concepts(sea_single_drift_concepts)

sea_three_drift_concepts = [
    SEA(function=1),
    SEA(function=3),
    SEA(function=4)
]

sea_three_drifts = recurrent_drift_for_sea_concepts(sea_three_drift_concepts)

sea_four_drift_concepts = [
    SEA(function=1),
    SEA(function=3),
    SEA(function=4)
]

sea_four_drifts = recurrent_drift_for_sea_concepts(sea_four_drift_concepts)
