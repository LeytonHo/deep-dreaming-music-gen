import soundfile as sf
import tensorflow as tf
from autoencoder import Autoencoder
from genre_classifier import Classifier, classifier_test
import pickle
import numpy as np
from genre_switcher import GenreSwitcher
from preprocess import INPUT_SIZE, SAMPLE_RATE

def train(model : Autoencoder, train_data, num_epochs, batch_size):
    """
	Runs through num_epochs epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_data: train data (all data for training) of shape (num_sentences, INPUT_SIZE)
	:param num_epochs: number of epochs to run
    :param batch_size: batch size
	:return: None
	"""
    for n in range(num_epochs):
        print("Epoch: ", n)
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i : i + batch_size]
            with tf.GradientTape() as tape:
                decoded = model.call(batch_data)
                batch_data = tf.slice(batch_data, (0,0,0), tf.shape(decoded))
                loss = model.loss(decoded, batch_data)
                total_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("Total Loss: ", total_loss)

    sf.write('input-train.wav', batch_data[0], SAMPLE_RATE)
    sf.write('output-train.wav', decoded[0], SAMPLE_RATE)

def test(model, test_data, batch_size):
    """
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_data: french test data (all data for testing) of shape (num_sentences, INPUT_SIZE)
	:returns: total loss of the test set
	"""
    total_loss = 0
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i : i + batch_size]
        decoded = model.call(batch_data)
        batch_data = tf.slice(batch_data, (0,0,0), tf.shape(decoded))
        loss = model.loss(decoded, batch_data)
        total_loss += loss

    # write first decoded thing to wav file! 
    sf.write('input-test.wav', batch_data[0], SAMPLE_RATE)
    sf.write('output-test.wav', decoded[0], SAMPLE_RATE)

    return total_loss

def get_train_and_test_data():
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

    y_train = genre_data[:train_tracks]
    y_test = y_test = genre_data[train_tracks:]
    ############# SHRINK FOR TESTING ###############################################
    SMOL = 10
    x_train = x_train[:SMOL]
    x_test = x_test[:SMOL]
    y_train = y_train[:SMOL]
    y_test = y_test[:SMOL]
    ################################################################################

    return x_train, x_test, y_train, y_test

def main():
    # set whether to load or compute models
    LOAD_AUTOENCODER = False
    LOAD_CLASSIFIER = False

    # load data
    x_train, x_test, y_train, y_test = get_train_and_test_data()

    ############ CREATE AUTOENCODER ########################################
    # create autoencoder
    autoencoder = Autoencoder(INPUT_SIZE)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
    autoencoder.compute_output_shape(input_shape=np.shape(x_train))
    autoencoder.build(np.shape(x_train))
    autoencoder.summary()

    ############ TRAIN AND SAVE AUTOENCODER #############################
    if not LOAD_AUTOENCODER:
        # train autoencoder
        num_epochs = 10
        train(autoencoder, x_train, num_epochs, 70)
        accuracy = test(autoencoder, x_test, 70)
        print("Autoencoder accuracy: ", accuracy)

        # Save autoencoder
        autoencoder.save_weights('saved_models/autoencoder_to_delete')

    ######## LOAD AUTOENCODER ##########################################
    if LOAD_AUTOENCODER:
        autoencoder.load_weights('saved_models/autoencoder_to_delete').expect_partial()

    ######## LOAD CLASSIFIER DATA #########################################
    genre_inputs_train = autoencoder.call(x_train)
    genre_inputs_test = autoencoder.call(x_test)

    print("SHAPE OF genre inputs train, test")
    print(genre_inputs_train.shape)
    print(genre_inputs_test.shape)

    # save autoencoder outputs
    with open('autoencoder_output.pickle', 'wb') as f:
        pickle.dump((genre_inputs_train, genre_inputs_test), f)

    ######## CREATE CLASSIFIER ############################################
    classifier = Classifier()
    classifier.compute_output_shape(input_shape=np.shape(genre_inputs_train))
    classifier.build(np.shape(genre_inputs_train))
    classifier.summary()

    ######## TRAIN AND SAVE CLASSIFIER ####################################
    if not LOAD_CLASSIFIER:
        x_train, x_test, y_train_one_hot, y_test_one_hot = classifier.pre_process(genre_inputs_train, genre_inputs_test, y_train, y_test)

        classifier.train(x_train, y_train_one_hot)
        accuracy = classifier_test(classifier, x_test, y_test_one_hot)
        print("Classifier accuracy: ", accuracy)

        classifier.save_weights('saved_models/classifier')

    ######## LOAD CLASSIFIER ##############################################
    if LOAD_CLASSIFIER:
        classifier.load_weights('saved_models/classifier')

    # switch genres
    # classifier = tf.keras.load_model('classifier')
    # autoencoder = tf.keras.load_model('autoencoder')
    # new_genre = "pop or something"
    # genre_switcher = GenreSwitcher(classifier, autoencoder, new_genre)
    # genre_switcher.compile(optimizer="adam")


if __name__ == '__main__':
	main()