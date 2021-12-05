import soundfile as sf
import tensorflow as tf
from autoencoder import Autoencoder
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

def main():
    autoencoder = Autoencoder(INPUT_SIZE)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

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
    
    autoencoder.build(np.shape(x_train))
    autoencoder.summary()

    train(autoencoder, x_train, 8, 70)
    accuracy = test(autoencoder, x_test, 70)
    print("Accuracy: ", accuracy)

    # switch genres
    classifier = tf.keras.load_model('classifier')
    autoencoder = tf.keras.load_model('autoencoder')
    new_genre = "pop or something"
    genre_switcher = GenreSwitcher(classifier, autoencoder, new_genre)
    genre_switcher.compile(optimizer="adam")


if __name__ == '__main__':
	main()