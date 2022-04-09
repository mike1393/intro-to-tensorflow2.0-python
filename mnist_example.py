import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

from models import functional_model, MyModel
from utils import display_samples, from_2D_to_3D, rescale_zero_to_one

# Sequential
seq_model = tf.keras.Sequential(
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

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # # Display samples
    if False:
        display_samples(x_train, y_train)
    # Preprocess for both training and testing
    x_train = from_2D_to_3D(rescale_zero_to_one(x_train))
    x_test = from_2D_to_3D(rescale_zero_to_one(x_test))
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)


    # The following line demonstrates how to create
    # a neural network model using three different ways.
    # 1. Sequential API
    # 2. Functional Model API
    # 3. Model Class
    # 2 and 3 can be found in ./models.py. 

    # # 1. Build the model using Sequential model API
    # model = seq_model
    # # 2. Build the model using Functional model API
    # model = functional_model()
    # # 3. Build the model using Model Class
    model = MyModel()

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    # model training
    model.fit(x_train, y_train, batch_size=64, epochs=3,validation_split=0.2)
    # model testing
    model.evaluate(x_test, y_test, batch_size=64)