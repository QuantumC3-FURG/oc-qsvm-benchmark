from sklearn.ensemble import IsolationForest


class Isolation_Forest:
    """
    Isolation Forest ensemble for anomaly detection.
    Anomalies are identified as instances with short average path lengths across the isolation trees.

    Parameters
    ----------
    params : list
        A list containing [random_state, n_jobs, contamination].

    Attributes
    ----------
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel CPU cores used during fitting.
    contamination : float
        Expected proportion of anomalies in the dataset.

    Examples
    --------
    >>> from core.Isolation_Forest import Isolation_Forest
    >>> model = Isolation_Forest([42, -1, 0.1])
    >>> model.fit(X_train)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self, params):
        self.params       = params
        self.random_state = params[0]
        self.n_jobs       = params[1]
        self.contamination = params[2]

    def _set_random(self, seed_value):
        import os
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        import random
        random.seed(seed_value)
        import numpy as np
        np.random.seed(seed_value)

    def _build_model(self):
        self._set_random(0)
        model = IsolationForest(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            contamination=self.contamination,
        )
        return model

    def fit(self, X):
        """
        Train the Isolation Forest on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Training input array of shape (samples, features).
        """
        self.model = self._build_model()
        self.model.fit(X)

    def predict(self, data):
        """
        Generate predictions using the trained model.
        Returns +1 for normal samples and -1 for anomalies.

        Parameters
        ----------
        data : numpy.ndarray
            Input array of shape (samples, features).

        Returns
        -------
        numpy.ndarray
            Prediction array with values +1 (normal) or -1 (anomaly).
        """
        return self.model.predict(data)
