import os
from pathlib import Path

import pandas as pd
from codecarbon import OfflineEmissionsTracker

from ADLClassifier.BaseClassifier import ADLClassifier


def record_emissions(adl_classifier: type(ADLClassifier)):

    class EmissionsRecorder(adl_classifier):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.emissions_recorder = OfflineEmissionsTracker(
                country_iso_code="DEU",
                log_level='ERROR',
                save_to_file=False
            )
            self.emissions_data = []

        def __str__(self):
            return f"{super().__str__()}WithEmissionTracking"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithEmissionTracking"

        # @track_emissions(offline=True, country_iso_code="DEU")
        def _train(self, instance):
            self.emissions_recorder.start()
            super()._train(instance)
            self.emissions_recorder.stop()
            self.emissions_data.append(self.emissions_recorder.final_emissions_data.values)

        @staticmethod
        def default_path() -> str:
            default = Path('results/codecarbon').absolute()
            default.mkdir(parents=True, exist_ok=True)
            return default.as_posix()

        def __del__(self):
            (pd.DataFrame(self.emissions_data)
             .to_csv(
                os.path.join(
                    os.environ.get("CODECARBON_OUTPUT_DIR", self.default_path()),
                    "emissions.csv"))
            )

    return EmissionsRecorder
