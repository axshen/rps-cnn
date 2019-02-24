import numpy as np
import matplotlib.pyplot as plt
import sys
import random

# ------------------------------------------------------------------------------
# READING DATA


def read_data(path, nmodel, num_classes, param):
    """
    read function for map data. For num_classes = 2, the param var doesn't matter.
    For num_classes = 1, decide whether to extract 'vrel' or 'rho'.
    """
    # update to terminal
    print('reading data from: ' + str(path))

    # constants used
    n_mesh = 50
    img_rows, img_cols = n_mesh, n_mesh
    n_mesh2 = n_mesh * n_mesh - 1
    n_mesh3 = n_mesh * n_mesh
    input_shape = (img_rows, img_cols, 1)

    # read files
    with open(path + '2dfv.dat') as f:
        lines = f.readlines()
    with open(path + '2dfvn.dat') as f:
        lines1 = f.readlines()
    X = np.zeros((nmodel, n_mesh3))
    y = np.zeros((nmodel, num_classes))

    # For 2D density map data
    ibin = 0
    jbin = -1
    for num, j in enumerate(lines):
        jbin = jbin + 1
        tm = j.strip().split()
        X[ibin, jbin] = float(tm[0])
        if jbin == n_mesh2:
            ibin += 1
            jbin =- 1

    # Y output
    ibin = 0
    for num, j in enumerate(lines1[1:]):
        tm = j.strip().split()
        if (num_classes == 1):
            if (param == 'v'):
                y[ibin, 0] = float(tm[0])
            elif (param == 'r'):
                y[ibin, 0] = float(tm[1])
            else:
                print("prediction parameter conversion applied")
                if (param == 'v2'):
                    y[ibin, 0] = v_to_v2(float(tm[0]))
                elif (param == 'P_RPS'):
                    y[ibin, 0] = float(tm[0])
                    y[ibin, 1] = float(tm[1])
                else:
                    print('Error in "param" variable (wot u choose)')
                    sys.exit()
        elif (num_classes == 2):
            y[ibin, 0] = float(tm[0])
            y[ibin, 1] = float(tm[1])
        ibin += 1

    # reshape
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)

    # return data
    return X, y


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


# ------------------------------------------------------------------------------
# VARIABLE CONVERSIONS


def v_to_v2(v):
    """
    convert v to v^2 (updated conversion)
    """
    v0 = v * (0.7 - 0.3) / (10) + 0.3
    v0_min2 = 0.3 ** 2
    v0_max2 = 0.7 ** 2
    v2 = 10 * (v0 ** 2 - v0_min2) / (v0_max2 - v0_min2)
    return v2


def RPS_P(v, rho):
    """
    Defining RPS pressure as potential prediction variable
    """
    rho0 = rho * (1 - 0.1) / (10) + 0.1
    v0 = v * (0.7 - 0.3) / (10) + 0.3
    P0 = rho0 * v0 ** 2
    P0_min = np.min(P0)
    P0_max = np.max(P0)
    P = 10 * (P0 - P0_min) / (P0_max - P0_min)
    return P


def mse(preds, truth):
    """
    Computing the appropriate mean squared error value for the test set predictions
    vs true values
    """
    preds = np.array(preds)
    truth = np.array(truth)
    difference = preds - truth
    mse = np.mean(difference ** 2)
    return mse

# ------------------------------------------------------------------------------
# VISUALISATION


def plot_galaxy(X, y, index):
    """
    Function to plot galaxies from 2D map data
    """
    galaxy_choice = np.matrix(X[index])
    plt.imshow(galaxy_choice, interpolation = "nearest")
    plt.title('Galaxy #%i, rho: %.2f' % (index, y[index]))
    plt.show()


def plot_histogram(var, name):
    """
    Simple plot of histogram of a chosen variable
    """
    # plotting histogram of values (for vrel and rho)
    plt.hist(var)
    plt.title("Histogram " + str(name))
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.show()


def plot_compare(pred, true):
    """
    Compare predictions with true values for paper figures
    """
    x_eq = np.arange(0,10,1)
    y_eq = np.arange(0,10,1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(true, pred, 'ro', alpha=0.01)
    plt.plot(x_eq, y_eq, color = 'black', dashes =[2, 2])
    plt.title("2D Density")
    plt.xlabel(r"$P'_{rps}$")
    plt.ylabel(r'$P_{rps}$')
    plt.show()


# ------------------------------------------------------------------------------
