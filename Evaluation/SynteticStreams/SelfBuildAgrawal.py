import copy

from capymoa.stream import Stream
from moa.streams.generators import AgrawalGenerator


class SelfBuildAgrawal(Stream):
    # there is no agrawal in capymoa, the docu lies (06.02.2025))

    def __init__(
            self,
            instance_random_seed: int = 1,
            function: int = 1,
            balance_classes: bool = False,
            peturb_fraction: float = 0.05,
    ):
        self.__init_args_kwargs__ = copy.copy(locals())  # save init args for recreation. not a deep copy to avoid unnecessary use of memory

        self.moa_stream = AgrawalGenerator()

        self.instance_random_seed = instance_random_seed
        self.function = function
        self.balance_classes = balance_classes
        self.peturb_fraction = peturb_fraction

        self.CLI = f"-i {instance_random_seed} -f {self.function} \
            {'-b' if self.balance_classes else ''} -p {self.peturb_fraction}"

        super().__init__(CLI=self.CLI, moa_stream=self.moa_stream)

    def __str__(self):
        attributes = [
            (
                f"instance_random_seed={self.instance_random_seed}"
                if self.instance_random_seed != 1
                else None
            ),
            f"function={self.function}",
            f"balance_classes={self.balance_classes}" if self.balance_classes else None,
            (
                f"peturb_fraction={self.peturb_fraction}"
                if self.peturb_fraction != 0.05
                else None
            ),
        ]
        non_default_attributes = [attr for attr in attributes if attr is not None]
        return f"Agriwal({', '.join(non_default_attributes)})"
