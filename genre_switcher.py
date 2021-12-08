import tensorflow as tf
import numpy as np
import pickle

# TODO:
# - Check that encoder, decoder, classifier are not being trained by the genre switcher
# - Check that the loss calculation is correct (correct loss? should be lists?)


class GenreSwitcher(tf.keras.Model):
    def __init__(self, classifier, autoencoder, desired_classification):
        super(GenreSwitcher, self).__init__()
        self.num_epochs = 5
        self.desired_classification = desired_classification
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.latent_vector = tf.Variable(np.zeros(self.autoencoder.input_size))

    def set_latent_vector(self, input):
        print(input.shape, self.autoencoder.input_size)
        latent_vector = self.autoencoder.encoder(input)
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
        optimizer = tf.keras.optimizers.Adam()
        self.set_latent_vector(original_song)

        for _ in range(self.num_epochs):
            with tf.GradientTape() as tape:
                classification = self.call()
                loss = self.loss(classification)
                print(loss)
        
            gradients = tape.gradients(self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # stop training if desired classification is reached

def main():
    # autoencoder = tf.keras.models.load_model('saved_models/autoencoder_to_delete', compile=False) 
    # classifier = tf.keras.models.load_model('saved_models/classifier', compile=False)


    # # genre_switcher = GenreSwitcher(classifier, autoencoder, 0)

    # ### LOAD DATA #############################################
    # with open('preprocessed.pickle', 'rb') as f:
    #     audio_data, sr_data, genre_data = pickle.load(f)

    # print(np.shape(audio_data))
    # audio_data = np.reshape(audio_data, (np.shape(audio_data)[0], np.shape(audio_data)[1], 1))
    # print(np.shape(audio_data))
    # total_tracks = np.shape(audio_data)[0]
    # train_tracks = int(total_tracks * 2 / 3)
    # print(train_tracks)
    # x_train = audio_data[:train_tracks]
    # x_test = audio_data[train_tracks:]
    # ###
    # SMOL = 10
    # x_train = x_train[:SMOL]
    # x_test = x_test[:SMOL]
    # ###
    # #############################################################
    # # print(autoencoder.encoder(x_train))
    # autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    # # autoencoder.build(np.shape(x_train))
    # autoencoder.call(x_train)
    pass

if __name__ == "__main__":
    main()

