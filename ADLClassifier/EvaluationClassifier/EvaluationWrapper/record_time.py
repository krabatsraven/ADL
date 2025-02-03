import time

from ADLClassifier import ADLClassifier


def record_time(adl_classifier: type(ADLClassifier)) -> type(ADLClassifier):
    class RecordTimeWrapper(adl_classifier):
        def __str__(self):
            return f"{super().__str__()}WithTimeRecords"

        @classmethod
        def name(cls) -> str:
            return f"{adl_classifier.name()}WithTimeRecords"

        def __init__(self):
            super().__init__()

            self.timings = {
                "covariance loop": 0,
            }

        def record_time(self, method, key: str, *args, **kwargs) -> None:
            time_start = time.time()
            result = method(*args, **kwargs)
            time_end = time.time()
            self.timings[key] += (time_end - time_start)
            return result

        @property
        def total_time_in_loop(self):
            return self.timings["covariance loop"]

    return RecordTimeWrapper
