import tensorflow as tf
from tensorflow.keras.layers import Dense
import pickle
import numpy as np


class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.num_genres = 8

        # Consider a conv architecture
        # TODO: Add a Flatten layer
        self.layer1 = Dense(1000, activation="tanh")
        self.layer2 = Dense(500, activation="relu")
        self.layer3 = Dense(250, activation="tanh")
        self.layer4 = Dense(50, activation="relu")
        self.layer5 = Dense(self.num_genres)

        self.batch_size = 100

    def call(self, x):
        output = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return output

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        total_loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        return tf.math.reduce_mean(total_loss)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    def train(model, train_inputs, train_labels):
        '''
        Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
        and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
        To increase accuracy, you may want to use tf.image.random_flip_left_right on your
        inputs before doing the forward pass. You should batch your inputs.
        
        :param model: the initialized model to use for the forward pass and backward pass
        :param train_inputs: train inputs (all inputs to use for training), 
        shape (num_inputs, width, height, num_channels)
        :param train_labels: train labels (all labels to use for training), 
        shape (num_labels, num_classes)
        :return: Optionally list of losses per batch to use for visualize_loss
        '''

        for i in range(0, len(train_inputs), model.batch_size):
            batch_images = train_inputs[i:min(len(train_inputs), i + model.batch_size)]
            batch_labels = train_labels[i:min(len(train_labels), i + model.batch_size)]

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            with tf.GradientTape() as tape:
                predictions = model.call(batch_images)
                loss = model.loss(predictions, batch_labels)
        
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    def pre_process(self, x_train, x_test, y_train, y_test):
        # Map genre IDs to indices as a procedure for converting to one-hot vectors
        genre_mapping = {
            21: 0, # Hip-Hop
            15: 1, # Electronic
            12: 2, # Rock
            1235: 3, # Instrumental
            2: 4, # International
            38: 5, # Experimental
            10: 6, # Pop
            17: 7, # Folk
        }

        # Indices of valid genre data (genre in the genre_mapping)
        y_train_indices = [i for i, genre in enumerate(y_train) if genre in genre_mapping]
        y_test_indices = [i for i, genre in enumerate(y_test) if genre in genre_mapping]

        # Get valid genre data
        y_train_genres = [genre_mapping[y_train[i]] for i in y_train_indices]
        y_test_genres = [genre_mapping[y_test[i]] for i in y_test_indices]

        # Get audio data associated with a valid genre
        x_train = np.array([x_train[i] for i in y_train_indices])
        x_test = np.array([x_test[i] for i in y_test_indices])

        # Convert genre data into one hot vectors
        y_train_one_hot = np.eye(self.num_genres)[y_train_genres]
        y_test_one_hot = np.eye(self.num_genres)[y_test_genres]

        return x_train, x_test, y_train_one_hot, y_test_one_hot


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    print("TEST FUNCTION")
    print(test_labels.shape)
    logits = model.call(test_inputs)
    return model.accuracy(logits, test_labels)

def main():
    classifier = Classifier()

    with open('preprocessed.pickle', 'rb') as f:
        audio_data, sr_data, genre_data = pickle.load(f)
    
    total_tracks = np.shape(audio_data)[0]
    train_tracks = int(total_tracks * 2 / 3)

    # Audio and genre training data
    x_train = audio_data[:train_tracks]
    y_train = genre_data[:train_tracks]

    # Audio and genre testing data
    x_test = audio_data[train_tracks:]
    y_test = genre_data[train_tracks:]

    x_train, x_test, y_train_one_hot, y_test_one_hot = classifier.pre_process(x_train, x_test, y_train, y_test)
    
    classifier.train(x_train, y_train_one_hot)
    accuracy = test(classifier, x_test, y_test_one_hot)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
	main()