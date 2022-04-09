import tensorflow as tf

from models import compare_models
from utils import from_2D_to_3D, rescale_zero_to_one, display_performance

"""
Model used in this example was built following the post on Kaggle.com
Post Name: How to choose CNN Architecture MNIST
Author: CHRIS DEOTTE
Link: https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
I am going to reconstruct some of the networks that he mentions and compare their performance
1. Basic: 32C5-P2-64C5-p2-D128-D10
2. Deeper CNN: 32C3-32C3-P2-64C3-64C3-P2-D128-D10
3. Basic w/Dropout: 32C5-P2-Dp40%-64C5-P2-Dp40%-D128-Dp40%-D10
4. Deeper CNN w/Dropout,BatchNormalization
    32C3-BN-32C3-BN-P2-Dp40%-64C3-BN-64C3-BN-P2-BN-Dp40%-D128-D10
The detail definition is in models.py
"""

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Preprocess for both training and testing
    x_train = from_2D_to_3D(rescale_zero_to_one(x_train))
    x_test = from_2D_to_3D(rescale_zero_to_one(x_test))
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)
    
    number_of_nets = 4
    history=[0]*number_of_nets
    names = ["32C5","32C3-32C3","32C5-Drop","32C3-32C3-BN-Drop"]
    epochs=10
    for i in range(number_of_nets):
        model = compare_models(i)
        # Compile the model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
        # model training
        history[i] = model.fit(x_train, y_train, batch_size=64, epochs=epochs,validation_split=0.2, verbose=0)
        print(f"CNN {names[i]}: Epochs={epochs}, Train accuracy={max(history[i].history['accuracy'])},Validation accuracy={max(history[i].history['val_accuracy'])}")
    
    line_styles = ['-','--',':','-.']
    display_performance(number_of_nets, history, names, line_styles)
