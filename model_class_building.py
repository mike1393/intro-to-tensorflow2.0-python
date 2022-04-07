import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
import numpy as np

# Model Class
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32,(3,3),activation='relu')
        self.conv2 = Conv2D(64,(3,3),activation='relu')
        self.pool1 = MaxPool2D()
        self.batch_norm1 = BatchNormalization()

        self.conv3 = Conv2D(128,(3,3),activation='relu')
        self.pool2 = MaxPool2D()
        self.batch_norm2 = BatchNormalization()

        self.global_avg_pool = GlobalAvgPool2D()
        self.d1 = Dense(64,activation='relu')
        self.d2 = Dense(10,activation='softmax')

    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)
        x = self.global_avg_pool(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

def preprocess(data):
    # Normalize the input to make the learning slightly more efficient
    # normalize data (img) from [0-255] to [0-1]
    data = data.astype("float")/255
    # Add the color channel dimension to our input data for Convnet
    return np.expand_dims(data, axis=-1)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Preprocess for both training and testing
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    # Since categorical_crossentropy requires label to be one_hot_encoded
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)
    # Build the model
    model = MyModel()
    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    # model training
    model.fit(x_train, y_train, batch_size=64, epochs=3,validation_split=0.2)
    # model testing
    model.evaluate(x_test, y_test, batch_size=64)