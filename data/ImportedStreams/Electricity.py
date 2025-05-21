from pathlib import Path

from capymoa.stream import ARFFStream

Electricity = ARFFStream(Path('data/electricity.arff').absolute().as_posix())
ElectricityTiny = ARFFStream(Path('data/electricity_tiny.arff').absolute().as_posix())