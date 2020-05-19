import cv2
import numpy as np


def read_images(file, n):
    """
    Read .dat file containing the simulated galaxy images.

    Args:
        f (str):        Path to image file
        n (int):        Number of images provided

    Returns:
        X (np.array):   Array of images (N, 50, 50, 1)
    """

    with open(file) as f:
        lines = f.readlines()

    n_mesh = 50
    images = np.zeros((n, n_mesh * n_mesh))

    ibin = 0
    jbin = -1
    for num, j in enumerate(lines):
        jbin = jbin + 1
        tm = j.strip().split()
        images[ibin, jbin] = float(tm[0])
        if jbin == (n_mesh - 1):
            ibin += 1
            jbin = - 1

    return images.reshape(n, n_mesh, n_mesh, 1)


def read_annots(file, n):
    """
    Read .dat file containing the annotations (v_rel, rho) corresponding
    to each simulated image.

    Args:
        f (str):        Path to image file
        n (int):        Number of images provided

    Returns:
        y (np.array):   Array of annotations (N, 50, 50, 2)
    """

    with open(file) as f:
        lines = f.readlines()

    n_params = 2
    annots = np.zeros((n, n_params))

    ibin = 0
    for num, j in enumerate(lines[1:]):
        tm = j.strip().split()
        annots[ibin, 0] = float(tm[0])
        annots[ibin, 1] = float(tm[1])
        ibin += 1

    return annots


def read_image(f):
    """
    Read an image (jpg/png) and resize to the expected size
    for prediction with trained CNN models (50x50). To be used for
    reading real images (HI column density) into appropriate shape for
    inference with model.

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
