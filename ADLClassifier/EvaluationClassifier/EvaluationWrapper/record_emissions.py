from codecarbon import track_emissions

from ADLClassifier import ADLClassifier


def record_emissions(adl_classifier: type(ADLClassifier)):

    class EmissionsRecorder(adl_classifier):
        def __str__(self):
            return f"{super().__str__()}WithEmissionTracking"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithEmissionTracking"

        @track_emissions(offline=True, country_iso_code="DEU")
        def _train(self, instance):
            super()._train(instance)

    return EmissionsRecorder
