import tensorflow as tf
from autoencoder import Autoencoder
import pickle
import numpy as np
from preprocess import INPUT_SIZE

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
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i : i + batch_size]
            with tf.GradientTape() as tape:
                decoded = model.call(batch_data)
                loss = model.loss(decoded, batch_data)
                print("Loss: ", loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_data):
    """
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_data: french test data (all data for testing) of shape (num_sentences, INPUT_SIZE)
	:returns: accuracy of the test set
	"""
    decoded = model.call(test_data)
    return model.accuracy(decoded, test_data)

def main():
    autoencoder = Autoencoder(INPUT_SIZE)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

    with open('preprocessed.pickle', 'rb') as f:
        audio_data, sr_data, genre_data = pickle.load(f)
    
    print(np.shape(audio_data))
    print(np.shape(sr_data))
    print(np.shape(genre_data))

    total_tracks = np.shape(audio_data)[0]
    train_tracks = int(total_tracks * 2 / 3)
    print(train_tracks)
    x_train = audio_data[:train_tracks]
    x_test = audio_data[train_tracks:]
    
    train(autoencoder, x_train, 10, 5)
    accuracy = test(autoencoder, x_test)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
	main()