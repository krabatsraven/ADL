from capymoa.stream.drift import AbruptDrift, DriftStream
from capymoa.stream.generator import SEA

sea_no_drift = DriftStream(
    stream=[
        SEA(function=1),
    ]
)

sea_single_drift = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
    ]
)

sea_drift_back_and_forth = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
        AbruptDrift(position=10000),
        SEA(function=1)
    ]
)

sea_three_drifts = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
        AbruptDrift(position=10000),
        SEA(function=4),
        AbruptDrift(position=15000),
        SEA(function=1),
    ]
)