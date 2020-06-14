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

    n_mesh = 50
    ibin = 0
    jbin = -1
    images = np.zeros((n, n_mesh * n_mesh))

    with open(file) as f:
        for line in f:
            jbin += 1
            tm = line.strip().split()
            images[ibin, jbin] = float(tm[0])
            if jbin == (n_mesh * n_mesh - 1):
                ibin += 1
                jbin = -1

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
    for _, j in enumerate(lines[1:]):
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


def reshape_y(y):
    """
    Reshape the output of RPS model prediction to the appropriate format
    for evaluation.

    Args:
        y (np.array):

    Returns:
        y (np.array)
    """
    pass


def split_annots(y):
    """
    Take array of annots (N, 2) and return v_rel and rho_igm.

    Args:
        y (np.array):       Array (N, 2) of annotations corresponding
                            to RPS simulations

    Returns:
        v_rel (np.array):   Array (N, ) of v_rel
        rho_igm (np.array): Array (N, ) of rho_igm
    """

    v_rel = y[:, 0]
    rho_igm = y[:, 1]
    return v_rel, rho_igm
