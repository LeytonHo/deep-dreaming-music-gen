import tensorflow as tf
import numpy as np
import pickle


class GenreSwitcher(tf.keras.Model):
    def __init__(self, classifier, autoencoder, desired_classification, input, learning_rate=0.01, num_epochs=1):
        super(GenreSwitcher, self).__init__()
        self.num_epochs = num_epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.desired_classification = desired_classification

        self.classifier = classifier
        self.autoencoder = autoencoder

        self.latent_vector = tf.Variable(self.autoencoder.encoder(input))

        # hold model weights fixed
        self.classifier.trainable = False
        self.autoencoder.trainable = False

    def call(self):
        """Predict the classification based upon the learned latent vector."""
        print("latent vector", self.latent_vector)
        return self.classifier(self.latent_vector)

    def loss_function(self, classification):
        """Calculate the deviation of the given classification from the desired."""
        total_loss = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(self.desired_classification, 8), classification)
        return tf.math.reduce_sum(total_loss)

    def compute_results(self):
        """Turn the learned latent vector into the music wave output."""
        return self.autoencoder.decoder(self.latent_vector)

    def train(self):
        for _ in range(self.num_epochs):
            with tf.GradientTape() as tape:
                classification = self.call()
                loss = self.loss_function(classification)
                print("loss:", loss)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
