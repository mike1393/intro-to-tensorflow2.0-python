import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
import numpy as np

# Model building:
# Sequential
# Functional model approach
def functional_model():
    input = Input((28,28,1))
    x = Conv2D(32,(3,3),activation='relu')(input)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(10,activation='softmax')(x)
    model = tf.keras.Model(inputs = input, outputs = x)
    return model

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
    model = functional_model()
    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    # model training
    model.fit(x_train, y_train, batch_size=64, epochs=3,validation_split=0.2)
    # model testing
    model.evaluate(x_test, y_test, batch_size=64)