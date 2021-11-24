import tensorflow as tf
from autoencoder import Autoencoder


def main():
    input_size = 330624
    autoencoder = Autoencoder(input_size)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

    # train the autoencoder
    autoencoder.fit(
        x_train,
        x_train,
        batch_size=100,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, x_test),
    )
