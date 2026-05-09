import math

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    ConvLSTM2D,
    Input,
    Layer,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam


class MSCRED:
    """
    Multi-Scale Convolutional Recurrent Encoder-Decoder for anomaly detection in time series.

    Multi-scale signature matrices are constructed to characterize system status across
    multiple temporal resolutions. A convolutional encoder captures inter-sensor correlation
    patterns, and an attention-based ConvLSTM network captures temporal dependencies.
    A convolutional decoder reconstructs the signature matrices; residuals from that
    reconstruction are used as anomaly scores.

    Parameters
    ----------
    params : list
        A list containing [sensor_n, scale_n, step_max].

    Attributes
    ----------
    model : Model
        The trained MSCRED model.

    Examples
    --------
    >>> from core.MSCRED import MSCRED
    >>> model = MSCRED([sensor_n, scale_n, step_max])
    >>> model.fit(X_train, Y_train)
    >>> predictions = model.predict(test_data)
    """

    def __init__(self, params):
        self.params = params

    def _build_model(self):
        self._set_random(0)

        class MyPadLayer(Layer):
            def __init__(self, paddings, **kwargs):
                super().__init__(**kwargs)
                self.paddings = paddings

            def call(self, inputs):
                return tf.pad(inputs, self.paddings)

        class MyAttentionLayer(Layer):
            def __init__(self, attention_fun, **kwargs):
                super().__init__(**kwargs)
                self.attention = attention_fun

            def call(self, inputs, **kwargs):
                return self.attention(inputs, **kwargs)

        class MyConcatLayer(Layer):
            def __init__(self, axis, **kwargs):
                super().__init__(**kwargs)
                self.axis = axis

            def call(self, inputs):
                return tf.concat(inputs, axis=self.axis)

        input_size = (
            self.params[2],
            self.params[0],
            self.params[0],
            self.params[1],
        )
        inputs = Input(input_size)

        if self.params[0] % 8 != 0:
            self.sensor_n_pad = (self.params[0] // 8) * 8 + 8
        else:
            self.sensor_n_pad = self.params[0]

        paddings = tf.constant([
            [0, 0],
            [0, 0],
            [0, self.sensor_n_pad - self.params[0]],
            [0, self.sensor_n_pad - self.params[0]],
            [0, 0],
        ])

        inputs_pad = MyPadLayer(paddings)(inputs)

        conv1 = TimeDistributed(Conv2D(
            filters=32, kernel_size=3, strides=1,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="conv1",
        ))(inputs_pad)

        conv2 = TimeDistributed(Conv2D(
            filters=64, kernel_size=3, strides=2,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="conv2",
        ))(conv1)

        conv3 = TimeDistributed(Conv2D(
            filters=128, kernel_size=2, strides=2,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="conv3",
        ))(conv2)

        conv4 = TimeDistributed(Conv2D(
            filters=256, kernel_size=2, strides=2,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="conv4",
        ))(conv3)

        convLSTM1 = ConvLSTM2D(
            filters=32, kernel_size=2, padding="same",
            return_sequences=True, name="convLSTM1",
        )(conv1)
        convLSTM1_out = MyAttentionLayer(self.attention)(convLSTM1, **{"koef": 1})

        convLSTM2 = ConvLSTM2D(
            filters=64, kernel_size=2, padding="same",
            return_sequences=True, name="convLSTM2",
        )(conv2)
        convLSTM2_out = MyAttentionLayer(self.attention)(convLSTM2, **{"koef": 2})

        convLSTM3 = ConvLSTM2D(
            filters=128, kernel_size=2, padding="same",
            return_sequences=True, name="convLSTM3",
        )(conv3)
        convLSTM3_out = MyAttentionLayer(self.attention)(convLSTM3, **{"koef": 4})

        convLSTM4 = ConvLSTM2D(
            filters=256, kernel_size=2, padding="same",
            return_sequences=True, name="convLSTM4",
        )(conv4)
        convLSTM4_out = MyAttentionLayer(self.attention)(convLSTM4, **{"koef": 8})

        deconv4 = Conv2DTranspose(
            filters=128, kernel_size=2, strides=2,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="deconv4",
        )(convLSTM4_out)
        deconv4_out = MyConcatLayer(axis=3)([deconv4, convLSTM3_out])

        deconv3 = Conv2DTranspose(
            filters=64, kernel_size=2, strides=2,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="deconv3",
        )(deconv4_out)
        deconv3_out = MyConcatLayer(axis=3)([deconv3, convLSTM2_out])

        deconv2 = Conv2DTranspose(
            filters=32, kernel_size=3, strides=2,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="deconv2",
        )(deconv3_out)
        deconv2_out = MyConcatLayer(axis=3)([deconv2, convLSTM1_out])

        deconv1 = Conv2DTranspose(
            filters=self.params[1], kernel_size=3, strides=1,
            kernel_initializer="glorot_uniform", padding="same",
            activation="selu", name="deconv1",
        )(deconv2_out)

        model = Model(
            inputs=inputs,
            outputs=deconv1[:, :self.params[0], :self.params[0], :],
        )
        return model

    def attention(self, outputs, koef):
        """
        Compute attention-weighted output over the temporal sequence.

        Each time step is weighted by its dot-product similarity with the final step.
        Weights are normalized via softmax before the weighted sum is applied.

        Parameters
        ----------
        outputs : tf.Tensor
            Output tensor from a ConvLSTM layer.
        koef : int
            Spatial downsampling factor relative to the first encoder scale.

        Returns
        -------
        tf.Tensor
            Attention-weighted output tensor.
        """
        attention_w = []
        for k in range(self.params[2]):
            attention_w.append(
                tf.reduce_sum(
                    tf.multiply(outputs[:, k], outputs[:, -1]), axis=(1, 2, 3)
                ) / self.params[2]
            )
        attention_w = tf.reshape(
            tf.nn.softmax(tf.stack(attention_w, axis=1)),
            [-1, 1, self.params[2]],
        )
        outputs = tf.reshape(
            outputs,
            [-1, self.params[2], tf.reduce_prod(outputs.shape.as_list()[2:])],
        )
        outputs = tf.matmul(attention_w, outputs)
        outputs = tf.reshape(outputs, [
            -1,
            math.ceil(self.sensor_n_pad / koef),
            math.ceil(self.sensor_n_pad / koef),
            32 * koef,
        ])
        return outputs

    def _set_random(self, seed_value):
        import os
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        import random
        random.seed(seed_value)
        import numpy as np
        np.random.seed(seed_value)
        import tensorflow as tf
        tf.random.set_seed(seed_value)

    def _loss_fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, X_train, Y_train, batch_size=200, epochs=25):
        """
        Train the MSCRED model on the provided data.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training input array of shape (samples, step_max, sensor_n, sensor_n, scale_n).
        Y_train : numpy.ndarray
            Training target array of shape (samples, sensor_n, sensor_n, scale_n).
        batch_size : int, optional
            Batch size used during training. Default is 200.
        epochs : int, optional
            Number of training epochs. Default is 25.
        """
        self.model = self._build_model()
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss=self._loss_fn)
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.8, patience=6, min_lr=0.000001, verbose=1
        )
        self.model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[reduce_lr],
        )

    def predict(self, data):
        """
        Generate reconstructions using the trained model.

        Parameters
        ----------
        data : numpy.ndarray
            Input array of shape (samples, step_max, sensor_n, sensor_n, scale_n).

        Returns
        -------
        numpy.ndarray
            Reconstructed signature matrix array of shape (samples, sensor_n, sensor_n, scale_n).
        """
        return self.model.predict(data)
