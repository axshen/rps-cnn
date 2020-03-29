import cv2
import numpy as np


def read_image(f):
    """
    Read an image (jpg/png) and resize to the expected size
    for prediction with trained CNN models (50x50). 

    Args:
        f (str):        Path to image file

    Returns:
        img (np.array): Image with correct dimensions (1,50,50,1)
                        and normalised so pixel values range from 0.0 -> 1.0
    """
    in_shape = (50, 50)
    out_shape = (50, 50, 1)
    img = cv2.imread(f, 0)
    img = np.reshape(cv2.resize(img, in_shape), out_shape)
    img = np.expand_dims(img / np.max(img), axis=0)
    return img