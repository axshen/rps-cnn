import cv2
import numpy as np


def read_images(path):
    """
    Read images for simulation map data. Requires path to the file (.dat)
    """
    n_mesh = 50
    img_rows, img_cols = n_mesh, n_mesh
    n_mesh2 = n_mesh * n_mesh - 1
    n_mesh3 = n_mesh * n_mesh

    with open(path) as f:
        lines = f.readlines()

    nmodel = int(len(lines) / (50 * 50))
    X = np.zeros((nmodel, n_mesh3))

    # Custom - for provided 2D density map data
    ibin = 0
    jbin = -1
    for num, j in enumerate(lines):
        jbin = jbin + 1
        tm = j.strip().split()
        X[ibin, jbin] = float(tm[0])
        if jbin == n_mesh2:
            ibin += 1
            jbin = - 1

    X = X.reshape(X.shape[0], img_rows, img_cols, 1)

    return X


def read_annotations(path):
    """
    Read labels for simulation map data.
    Produces array (nmodel, 2) for the labels for each image.
    """
    nclasses = 2

    with open(path) as f:
        lines = f.readlines()
    nmodel = int(len(lines)) - 1
    y = np.zeros((nmodel, nclasses))

    # Y output (index 0: vrel, index 1: rho)
    ibin = 0
    for num, j in enumerate(lines[1:]):
        tm = j.strip().split()
        y[ibin, 0] = float(tm[0])
        y[ibin, 1] = float(tm[1])
        ibin += 1

    return y


def read_predictions(file):
    """
    Read predictions from gcloud compute trained keras models on
    image input data (should be a dat file)
    """
    output = []
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        pred = float(line)
        output.append(pred)
    f.close()
    return output


def read_image(filename):
    """
    Read an image file and convert to appropriate
    map input image.
    """
    img = cv2.imread(filename, 0)
    img = np.reshape(cv2.resize(img, (50, 50)), (50, 50, 1))
    return img / np.max(img)


def write_predictions(path, preds, NUM_CLASSES):
    """
    Write model predictions to file
    """
    f = open(path + "predictions.dat", 'w')
    for i in range(len(preds)):
        if (NUM_CLASSES == 1):
            f.write(str(preds[i][0]) + '\n')
        else:
            f.write(str(preds[i][0]) + ',' + str(preds[i][1]) + '\n')
    f.close()
