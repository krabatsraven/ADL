from capymoa.stream.drift import DriftStream, AbruptDrift
from capymoa.stream.generator import SEA

from Evaluation.SynteticStreams.SelfBuildAgrawal import SelfBuildAgrawal

simple_sea_single_drift = DriftStream(
    stream=[
        SEA(function=1),
        AbruptDrift(position=5000),
        SEA(function=3),
    ]
)

simple_agraval_single_drift = DriftStream(
    stream=[
        SelfBuildAgrawal(function=1),
        AbruptDrift(position=5000),
        SelfBuildAgrawal(function=3),
    ]
)

simple_agraval_drift_back_and_forth = DriftStream(
    stream=[
        SelfBuildAgrawal(function=1),
        AbruptDrift(position=5000),
        SelfBuildAgrawal(function=3),
        AbruptDrift(position=5000),
        SelfBuildAgrawal(function=1)
    ]
)

simple_agraval_three_drifts = DriftStream(
    stream=[
        SelfBuildAgrawal(function=1),
        AbruptDrift(position=5000),
        SelfBuildAgrawal(function=3),
        AbruptDrift(position=10000),
        SelfBuildAgrawal(function=5),
        AbruptDrift(position=15000),
        SelfBuildAgrawal(function=1),
    ]
)