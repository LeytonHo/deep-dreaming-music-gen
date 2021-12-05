import tensorflow as tf

# TODO:
# - Check that encoder, decoder, classifier are not being trained by the genre switcher
# - Check that the loss calculation is correct (correct loss? should be lists?)


class GenreSwitcher(tf.keras.Model):
    def __init__(self, classifier, autoencoder, desired_classification):
        super(GenreSwitcher, self).__init__()
        self.desired_classification = desired_classification
        self.classifier = classifier
        self.autoencoder = autoencoder
        initial_latent_vector = self.autoencoder.encoder(
            # random vector with results normally distributed between -2 and 2
            tf.random.truncated_normal([self.autoencoder.input_size])
        )
        self.latent_vector = tf.Variable(initial_latent_vector)

    def call(self, latent_vector):
        """Predict the classification based upon the learned latent vector."""
        classification = self.classifier(latent_vector)

        return classification

    def loss(self, classification):
        """Calculate the deviation of the given classification from the desired."""
        return tf.keras.losses.sparse_categorical_crossentropy(
            self.desired_classification, classification
        )

    def compute_results(self):
        """Turn the learned latent vector into the music wave output."""
        return self.autoencoder.decoder(self.latent_vector)
