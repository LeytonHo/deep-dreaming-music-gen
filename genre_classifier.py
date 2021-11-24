import tensorflow as tf
from tensorflow.keras.layers import Dense


class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = Dense(100, activation="relu")
        self.layer2 = Dense(50, activation="relu")
        self.layer3 = Dense(8)

    def call(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)

        return output
