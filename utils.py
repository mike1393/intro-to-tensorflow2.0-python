import numpy as np
import matplotlib.pyplot as plt

def display_performance(number_of_nets, history, names, line_styles):
    # PLOT ACCURACIES
    plt.figure(figsize=(15,5))
    for i in range(number_of_nets):
        plt.plot(history[i].history['val_accuracy'],linestyle=line_styles[i])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0.98,1])
    plt.show()

def display_samples(samples, labels, row=3, col=3,figure_size=(10,10)):
    plt.figure(figsize=figure_size)
    for i in range(16):
        idx = np.random.randint(0,samples.shape[0]-1)
        img = samples[idx]
        label = labels[idx]
        plt.subplot(row,col,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()

def rescale_zero_to_one(data):
    # normalize data (img) from [0-255] to [0-1]
    return data.astype("float")/255

def from_2D_to_3D(data):
    # Add the color channel dimension to our input data for Convnet
    return np.expand_dims(data, axis=-1)