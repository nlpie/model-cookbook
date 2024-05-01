import dataclasses
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn


def create_model(config_path, weights_path=None):
    """The entry point for creating the model.

    Args:
        config_path: Path to the configuration file for the model
        weights_path: Path to the saved weights for the model

    Returns:
        The loaded model.

    """
    with open(os.fspath(config_path)) as f:
        config = json.load(f)
    model = MyModel(config)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


class MyModel(nn.Module):
    """Example parameterized simple MLP with dynamic number of layers. It would be called with a config like:

    >>> config = {'input_features': 10, 'hidden_features': [100, 100]}
    >>> model = MyModel(config)

    The config can be saved or loaded to json

    >>> import json
    >>> # loading
    >>> with open('config.json') as f:
    >>>     config = json.load(f)
    >>> # saving
    >>> with open('config.json', 'w') as f:
    >>>     json.dump(config, f)

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        prev = config['input_features']
        layers = []
        for out_dim in config['hidden_features']:
            layers.append(nn.Linear(prev, out_dim))
            prev = out_dim
        layers.append(nn.Linear(prev, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, df):
        """Predicts the examples in a pandas DataFrame and returns probabilities of a positive classification.

        Args:
            df: The examples DataFrame

        DataFrame Columns:
            # Here we're documenting what columns are expected in the DataFrame and their data types
            REASON_FOR_ADMITTANCE: nullable string, an example categorical feature
            AGE: non-null int64, an example integer feature
            SP02_MAX: nullable float64, an example float feature

        Returns:
            Tensor of positive probability for all examples

        """
        logits = self(transform(df))
        return nn.functional.sigmoid(logits)


def transform(df):
    """Transforms a dataframe of example(s) into a tensor for prediction. If this was an image model, you would
    convert the image data from a path to a tensor here. You would use this function both prior to training and during
    inference.

    Args:
        df: A pandas DataFrame of one or more (in case of Batch processing, multiple images, etc) examples.

    Returns:
        Torch tensor representation

    """
    # Example converting a categorical column (string type) to one hots, reasons that aren't A, B, or C will be
    # converted to NaN
    cat = df['REASON_FOR_ADMITTANCE'].astype(pd.CategoricalDtype(categories=["A", "B", "C"]))
    one_hot = pd.get_dummies(cat, prefix='Reason', dummy_na=True)  # Also adds na column
    df = df.drop('Category', axis=1).join(one_hot)

    # Adding an IS_NA column for nullable feature
    df['SP02_MAX_NaN'] = df['SPO2_MAX'].isna()

    # Convert to a tensor, selecting and ordering the model inputs
    features = ["Reason_A", "Reason_B", "Reason_C", "Reason_NaN", "AGE", "SPO2_MAX", "SP02_MAX_NaN"]
    return torch.tensor(df[features].values.astype(np.float32))
