import tensorflow as tf
from tensorflow.keras.layers import (
    MaxPooling1D,
    Flatten,
    Reshape,
    Dense,
    Conv1D,
    Conv2D,
    Conv1DTranspose,
    Conv2DTranspose,
)


class Autoencoder(tf.keras.Model):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                Conv1D(256, 32, strides=8, activation="relu"),
                MaxPooling1D(2),
                Conv1D(32, 16, strides=8, activation="relu"),
                MaxPooling1D(2),
            ],
            'encoder'
        )
        self.decoder = tf.keras.Sequential(
            [
                Conv1DTranspose(32, 32, strides=16, activation="relu"),
                Conv1DTranspose(1, 32, strides=16, activation="tanh"),
            ],
            'decoder'
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # multiply by 2 because tanh between -1 and 1 and amplitude -2 to 2
        return 2 * decoded

    def accuracy(self, x_pred, x_true):
        return self.loss(x_pred, x_true)  # TODO: fix
