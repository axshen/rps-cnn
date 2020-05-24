import numpy as np
import matplotlib.pyplot as plt


def show_image(X, index):
    """
    Visualise an image from the training dataset.

    Args:
        X (np.array):   Array (n, 50, 50, n_channels) of images
        index (int):    Index of image to visualise.
    """

    img = np.matrix(X[index])
    plt.imshow(img, interpolation="nearest")
    plt.show()
