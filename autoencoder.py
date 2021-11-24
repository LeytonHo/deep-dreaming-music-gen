import tensorflow as tf
from tensorflow.keras.layers import Dense


class Autoencoder(tf.keras.Model):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = 420
        self.latent_dim = 69
        self.encoder = tf.keras.Sequential(
            [
                Dense(self.hidden_dim, activation="relu"),
                Dense(self.hidden_dim, activation="relu"),
                Dense(self.hidden_dim, activation="relu"),
                Dense(self.latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(self.hidden_dim, activation="relu"),
                Dense(self.hidden_dim, activation="relu"),
                Dense(self.hidden_dim, activation="relu"),
                Dense(self.input_size, activation="sigmoid"),  # maybe not sigmoid??
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
