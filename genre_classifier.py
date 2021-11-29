import tensorflow as tf
from tensorflow.keras.layers import Dense
import pickle
import numpy as np


class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = Dense(100, activation="relu")
        self.layer2 = Dense(50, activation="relu")
        self.layer3 = Dense(8)

        self.batch_size = 10

    def call(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)

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

    logits = model.call(test_inputs)
    print("LOGITS: ")
    print(logits)
    print("test_labels: ")
    print(test_labels)
    return model.accuracy(logits, test_labels)

def main():
    classifier = Classifier()

    with open('preprocessed.pickle', 'rb') as f:
        audio_data, sr_data, genre_data = pickle.load(f)
    
    total_tracks = np.shape(audio_data)[0]
    train_tracks = int(total_tracks * 2 / 3)
    x_train = audio_data[:train_tracks]
    x_test = audio_data[train_tracks:]

    y_train = genre_data[:train_tracks]
    y_train = np.expand_dims(y_train, axis=1)
    y_test = genre_data[train_tracks:]
    y_test = np.expand_dims(y_test, axis=1)

    # TODO: Convert labels to one-hot vectors / use another approach
    # Figure out a mapping between genre # (i.e. 103) to one-hot vector index for this

    classifier.train(x_train, y_train)
    accuracy = test(classifier, x_test, y_test)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
	main()