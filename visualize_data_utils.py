import numpy as np
import matplotlib.pyplot as plt

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