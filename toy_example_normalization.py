import torch
import re
import numpy as np
from pathlib import Path
from capymoa.stream import ARFFStream
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from Evaluation._config import STREAM_STRINGS
from Evaluation.config_handling import config_to_stream


class StreamingStandardScaler(StandardScaler):

    def fit(self, X, y=None, sample_weight=None):
        return self

    def transform(self, X, copy=None):
        self.partial_fit(X)
        return super().transform(X, copy)

for stream_name in STREAM_STRINGS:
    stream = config_to_stream(stream_name)

    transformers = [
        elem
        for elem in
        [
            (OneHotEncoder(categories=[torch.arange(len(stream.schema.get_moa_header().attribute(i).getAttributeValues()), dtype=torch.float32)]), [i])
            if stream.schema.get_moa_header().attribute(i).isNominal()
            else (StreamingStandardScaler(), [i])
            if stream.schema.get_moa_header().attribute(i).isNumeric()
            else None
            for i in range(stream.schema.get_num_attributes())
        ]
        if elem is not None
    ]

    column_trans = make_column_transformer(
        *transformers,
        remainder='passthrough',
        sparse_threshold=0,
    )
    transformer = column_trans
    i = 0
    while i < 10:
        current_instance = stream.next_instance()
        if len(current_instance.x.shape) < 2:
            x = current_instance.x[np.newaxis, ...]
        else:
            x = current_instance.x

        if i == 0:
            transformer.fit(x)
        transformed_nominal = transformer.transform(x)
        out_tensor = torch.from_numpy(transformed_nominal).to(dtype=torch.float)
        print(x)
        print(out_tensor)
        print()

        i += 1