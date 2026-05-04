from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Conv_AE:
    """
    Reconstruction-based convolutional autoencoder for anomaly detection in time series.
    Anomaly scores are derived from per-sample reconstruction error.

    No initialization parameters are required.

    Attributes
    ----------
    model : Sequential
        The trained convolutional autoencoder.

    Examples
    --------
    >>> from core.Conv_AE import Conv_AE
    >>> model = Conv_AE()
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self):
        self._set_random(0)

    def _set_random(self, seed_value):
        import os
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        import random
        random.seed(seed_value)
        import numpy as np
        np.random.seed(seed_value)
        import tensorflow as tf
        tf.random.set_seed(seed_value)

    def _build_model(self):
        n_features = self.shape[2]
        model = Sequential([
            Input(shape=(self.shape[1], n_features)),
            Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
            Dropout(rate=0.2),
            Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
            Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
            Dropout(rate=0.2),
            Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
            Conv1DTranspose(filters=n_features, kernel_size=7, padding="same"),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    def fit(self, data):
        """
        Train the convolutional autoencoder on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Training input array of shape (samples, timesteps, features).
        """
        self.shape = data.shape
        self.model = self._build_model()
        self.model.fit(
            data, data,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
            ],
        )

    def predict(self, data):
        """
        Generate reconstructions using the trained model.

        Parameters
        ----------
        data : numpy.ndarray
            Input array of shape (samples, timesteps, features).

        Returns
        -------
        numpy.ndarray
            Reconstructed output array.
        """
        return self.model.predict(data)
