import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Dropout, Flatten
"""
The following model was built following the post on Kaggle.com
Post Name: How to choose CNN Architecture MNIST
Author: CHRIS DEOTTE
Link: https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
I am going to reconstruct some of the networks that he mentions and compare their performance
1. Basic: 32C5-P2-64C5-p2-D128-D10
2. Deeper CNN: 32C3-32C3-P2-64C3-64C3-P2-D128-D10
3. Basic w/Dropout: 32C5-P2-Dp40%-64C5-P2-Dp40%-D128-Dp40%-D10
4. Deeper CNN w/Dropout,BatchNormalization
    32C3-BN-32C3-BN-P2-Dp40%-64C3-BN-64C3-BN-P2-BN-Dp40%-D128-D10
"""
def compare_models(idx):
    input = Input((28,28,1))
    if idx%2 == 0:
        x = Conv2D(32,kernel_size=5,activation='relu')(input)
    else:
        x = Conv2D(32,kernel_size=3,activation='relu')(input)
        x = BatchNormalization()(x) if idx==3 else x
        x = Conv2D(32,kernel_size=3,activation='relu')(x)
        x = BatchNormalization()(x) if idx==3 else x
    x = MaxPool2D()(x)
    x = BatchNormalization()(x) if idx==3 else x
    x = Dropout(0.4)(x) if idx >= 2 else x
    if idx%2 == 0:
        x = Conv2D(64,kernel_size=5,activation='relu')(x)
    else:
        x = Conv2D(64,kernel_size=3,activation='relu')(x)
        x = BatchNormalization()(x) if idx==3 else x
        x = Conv2D(64,kernel_size=3,activation='relu')(x)
        x = BatchNormalization()(x) if idx==3 else x
    x = MaxPool2D()(x)
    x = BatchNormalization()(x) if idx==3 else x
    x = Dropout(0.4)(x) if idx >= 2 else x
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.4)(x) if idx >= 2 else x
    x = Dense(10,activation='softmax')(x)
    model = tf.keras.Model(inputs = input, outputs = x)
    return model
"""
The following model was built following the post on https://aifee.teachable.com
Lecture Name: Introduction to Tensorflow 2 for Computer Vision
Author: Nour Islam Mokhtari
I am going to build a model using two methods
1. Functional Model API
2. Model Subclassing
"""
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

# Model Subclassing
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