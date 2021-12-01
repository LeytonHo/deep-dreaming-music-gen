import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv1DTranspose, Conv2DTranspose


class Autoencoder(tf.keras.Model):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim1 = 1000
        self.hidden_dim2 = 1500
        self.latent_dim = 1000
        self.encoder = tf.keras.Sequential(
            [
                Conv1D(1, 15),
                Dense(self.hidden_dim1, activation="relu"),
                Dense(self.hidden_dim2, activation="relu"),
                Dense(self.latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(self.hidden_dim2, activation="relu"),
                Dense(self.hidden_dim1, activation="relu"),
                Conv1DTranspose(1, 15),
                Dense(self.input_size, activation="tanh"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return 2 * decoded

    def loss(self, x_pred, x_true):
        return tf.reduce_sum(tf.keras.losses.MeanSquaredError(x_true, x_pred))
    
    def accuracy(self, x_pred, x_true):
        return self.loss(x_pred, x_true) # TODO: fix
