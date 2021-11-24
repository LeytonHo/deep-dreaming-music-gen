import tensorflow as tf
from autoencoder import Autoencoder


def main():
    input_size = 330624
    autoencoder = Autoencoder(input_size)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
