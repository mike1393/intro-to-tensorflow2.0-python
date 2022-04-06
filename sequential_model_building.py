import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
import numpy as np

# Model building:
# Sequential
model = tf.keras.Sequential(
    [
        Input((28,28,1)),
        Conv2D(32,(3,3),activation='relu'),
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64,activation='relu'),
        Dense(10,activation='softmax')
    ]
)

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
    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    # model training
    model.fit(x_train, y_train, batch_size=64, epochs=3,validation_split=0.2)
    # model testing
    model.evaluate(x_test, y_test, batch_size=64)