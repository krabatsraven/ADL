from capymoa.stream.drift import DriftStream, AbruptDrift
from capymoa.stream.generator import AgrawalGenerator

agrawal_no_drift = DriftStream(
    stream=[AgrawalGenerator(classification_function=1)]
)


agrawal_single_drift = DriftStream(
    stream=[
        AgrawalGenerator(classification_function=1),
        AbruptDrift(position=5000),
        AgrawalGenerator(classification_function=3),
    ]
)

agrawal_drift_back_and_forth = DriftStream(
    stream=[
        AgrawalGenerator(classification_function=1),
        AbruptDrift(position=5000),
        AgrawalGenerator(classification_function=3),
        AbruptDrift(position=10000),
        AgrawalGenerator(classification_function=1)
    ]
)

agrawal_three_drifts = DriftStream(
    stream=[
        AgrawalGenerator(classification_function=1),
        AbruptDrift(position=5000),
        AgrawalGenerator(classification_function=3),
        AbruptDrift(position=10000),
        AgrawalGenerator(classification_function=4),
        AbruptDrift(position=15000),
        AgrawalGenerator(classification_function=1),
    ]
)