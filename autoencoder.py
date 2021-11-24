import tensorflow as tf
from tensorflow.keras.layers import Dense


class Autoencoder(tf.keras.Model):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim1 = 50000
        self.hidden_dim2 = 10000
        self.hidden_dim3 = 1000
        self.latent_dim = 420
        self.encoder = tf.keras.Sequential(
            [
                Dense(self.hidden_dim1, activation="relu"),
                Dense(self.hidden_dim2, activation="relu"),
                Dense(self.hidden_dim3, activation="relu"),
                Dense(self.latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(self.hidden_dim3, activation="relu"),
                Dense(self.hidden_dim2, activation="relu"),
                Dense(self.hidden_dim1, activation="relu"),
                Dense(self.input_size, activation="tanh"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return 2 * decoded
