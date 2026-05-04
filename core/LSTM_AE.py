from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed


class LSTM_AE:
    """
    Sequence-to-sequence LSTM autoencoder for anomaly detection in time series.
    Anomaly scores are derived from per-sample reconstruction error.

    Parameters
    ----------
    params : list
        A list containing [EPOCHS, BATCH_SIZE, VAL_SPLIT].

    Attributes
    ----------
    model : Model
        The trained LSTM autoencoder.

    Examples
    --------
    >>> from core.LSTM_AE import LSTM_AE
    >>> model = LSTM_AE([100, 32, 0.1])
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self, params):
        self.params = params

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
        self._set_random(0)
        inputs  = Input(shape=(self.shape[1], self.shape[2]))
        encoded = LSTM(100, activation="relu")(inputs)
        decoded = RepeatVector(self.shape[1])(encoded)
        decoded = LSTM(100, activation="relu", return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.shape[2]))(decoded)
        model   = Model(inputs, decoded)
        model.compile(optimizer="adam", loss="mae", metrics=["mse"])
        return model

    def fit(self, X):
        """
        Train the LSTM autoencoder on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Training input array of shape (samples, timesteps, features).
        """
        self.shape = X.shape
        self.model = self._build_model()
        early_stopping = EarlyStopping(patience=5, verbose=0)
        self.model.fit(
            X, X,
            validation_split=self.params[2],
            epochs=self.params[0],
            batch_size=self.params[1],
            verbose=0,
            shuffle=False,
            callbacks=[early_stopping],
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
