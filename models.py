import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

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