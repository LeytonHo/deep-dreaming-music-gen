import tensorflow as tf
import numpy as np
import pickle

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

    def set_latent_vector(self, latent_vector):
        self.latent_vector = tf.Variable(latent_vector)

    def call(self):
        """Predict the classification based upon the learned latent vector."""
        classification = self.classifier(self.latent_vector)

        return classification

    def loss(self, classification):
        """Calculate the deviation of the given classification from the desired."""
        return tf.keras.losses.sparse_categorical_crossentropy(
            self.desired_classification, classification
        )

    def compute_results(self):
        """Turn the learned latent vector into the music wave output."""
        return self.autoencoder.decoder(self.latent_vector)

    def train(self, original_song):
        # TODO
        latent_vector = self.autoencoder.encoder(original_song)
        self.set_latent_vector(latent_vector)

        with tf.GradientTape() as tape:
            classification = self.call()
            loss = self.loss(classification)

def main():
    autoencoder = tf.keras.models.load_model('saved_models/autoencoder_to_delete', compile=False) 
    classifier = tf.keras.models.load_model('saved_models/classifier', compile=False)


    # genre_switcher = GenreSwitcher(classifier, autoencoder, 0)

    ### LOAD DATA #############################################
    with open('preprocessed.pickle', 'rb') as f:
        audio_data, sr_data, genre_data = pickle.load(f)

    print(np.shape(audio_data))
    audio_data = np.reshape(audio_data, (np.shape(audio_data)[0], np.shape(audio_data)[1], 1))
    print(np.shape(audio_data))
    total_tracks = np.shape(audio_data)[0]
    train_tracks = int(total_tracks * 2 / 3)
    print(train_tracks)
    x_train = audio_data[:train_tracks]
    x_test = audio_data[train_tracks:]
    ###
    SMOL = 10
    x_train = x_train[:SMOL]
    x_test = x_test[:SMOL]
    ###
    #############################################################
    # print(autoencoder.encoder(x_train))
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    # autoencoder.build(np.shape(x_train))
    autoencoder.call(x_train)

if __name__ == "__main__":
    main()

