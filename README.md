# Introduction to Tensorflow2.0 with Python
### About this Repo
* The [first part](https://github.com/mike1393/intro-to-tensorflow2.0-python/edit/main/README.md#part-1-learning-tensorflow20-keras) of the project was to learn [Tensorflow2.0](https://www.tensorflow.org/tutorials).
I followed the tutorial given by [Nour Islam Mokhtari](https://aifee.teachable.com/).
So, i you are interested in the work or want to learn Tensorflow, make sure to check him out.:grin:<br>
* The [second part](https://github.com/mike1393/intro-to-tensorflow2.0-python/edit/main/README.md#part-2-model-analysis) of the repo is to reproduce the work given by [CHRIS DEOTTE](https://www.kaggle.com/cdeotte), where he posted a great article about [How to choose CNN Architecture MNIST](https://www.kaggle.com/code/cdeotte/how-to-choose-cnn-architecture-mnist/notebook#5.-Advanced-features):point_left: on [Kaggle.com](https://www.kaggle.com/).<br> 
Here I chose several models mentioned in this article and create a comparison chart in regard of the validation accuracy. For more details, make sure to check it out if you are interested.:thumbsup:<br>

### Dataset
Both the first and second part of the project uses MNIST dataset, the very same dataset I used in the other repo, [simple-neural-network-python](https://github.com/mike1393/simple-neural-network-python). In that repo, I built a fully connected neural network from scratch using python. Feel free to check it out if you are interested.:raised_hands:<br>

### Part 1. Learning Tensorflow2.0, Keras
The first goal of this project is to learn how to use Tensorflow2.0 to classify handwritten digits.
I demonstrated three methods to build a model in this framework. For detail implementation, please see [./models.py](https://github.com/mike1393/intro-to-tensorflow2.0-python/blob/main/models.py)
1. [Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) (The implementation can be found in [./mnist_example.py](https://github.com/mike1393/intro-to-tensorflow2.0-python/blob/main/mnist_example.py))
2. [Functional Model API](https://www.tensorflow.org/guide/keras/functional)
3. [Model Subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models)

### Part 2. Model Analysis
The second goal was to reproduce the work of Chris Deotte. In his post, he shared the strategy of finding the best CNN architecture for MNIST.
Among all the architectures he mentioned, I chose four of them and plot the validation accuracy performance of 5 and 10 epochs respectively.
Here are the models,
1. Basic: 32C5-P2-64C5-p2-D128-D10
2. Deeper CNN: 32C3-32C3-P2-64C3-64C3-P2-D128-D10
3. Basic w/Dropout: 32C5-P2-Dp40%-64C5-P2-Dp40%-D128-Dp40%-D10
4. Deeper CNN w/Dropout,BatchNormalization: 32C3-BN-32C3-BN-P2-Dp40%-64C3-BN-64C3-BN-P2-BN-Dp40%-D128-D10
<br>(32C5 means a convolution layer with 32 feature maps using a 5x5 filter and stride 1. P2 means max pooling using 2x2 filter and stride 2. BN means BatchNormalizer.
 Dp40% means 40% Dropout.)<br>
 To prevent repeating myself and for a cleaner code, I chose functional model API to build the model.(Detail implementation in [./models.py](https://github.com/mike1393/intro-to-tensorflow2.0-python/blob/main/models.py)) and the rest of the code can be found in [./mnist_kaggle_best.py](https://github.com/mike1393/intro-to-tensorflow2.0-python/blob/main/mnist_kaggle_best.py).
 #### Result
   * Trains for 10 epochs<br>
  ![10 epochs](https://github.com/mike1393/intro-to-tensorflow2.0-python/blob/main/result/epochs_10.png)
  * By comparing model 1(32C5) and model 2(32C3-32C3), we can see how adding convolution layers can effect the accuracy.
  * By comparing model 1(32C5) and model 3(32C5-Drop), we can see how adding dropout layers can effect the performance.
  * Finally, in model 4(32C3-32C3-BN-Drop), we get the best performance by adding more convolution layers, dropout layers, and batch normalization layers.

