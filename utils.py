import random
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


class inout:
    def __init__(self):
        pass

    def read_file(file_path, nmodel, num_classes):
        """
        Method for reading additional files provided.
        """
        n_mesh = 50
        img_rows, img_cols = n_mesh, n_mesh
        n_mesh2 = n_mesh * n_mesh - 1
        n_mesh3 = n_mesh * n_mesh
        input_shape = (img_rows, img_cols, 1)

        X = np.zeros((nmodel, n_mesh3))

        with open(file_path) as f:
            lines_X = f.readlines()

        # For 2D density map data
        ibin = 0
        jbin = -1
        for num, j in enumerate(lines_X):
            jbin = jbin + 1
            tm = j.strip().split()
            X[ibin, jbin] = float(tm[0])
            if jbin == n_mesh2:
                ibin += 1
                jbin = - 1

        X = X.reshape(X.shape[0], img_rows, img_cols, 1)

        return X

    def read_data(path, nmodel, num_classes, param):
        """
        read function for map data. For num_classes = 2, the param var doesn't matter.
        For num_classes = 1, decide whether to extract 'vrel' or 'rho'. Param = v, r, v2
        or P_RPS.
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
                jbin = - 1

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


class conversion:
    def __init__(self):
        pass

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


class visualisation:
    def __init__(self):
        pass

    def grid_histogram(data, plot_title, titles, x_labels, y_labels, save, filename):
        """
        Plot histogram of data.
        """
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        subtitle_font = {'weight': 'normal', 'size': 12}
        title_font = {'weight': 'bold', 'size': 14}
        text_properties = {'verticalalignment': 'top', 'horizontalalignment': 'left'}
    
        n = len(data)
        dim1 = math.ceil(math.sqrt(n))
        dim2 = math.ceil(n / dim1)
        bins = 40

        plt.figure()
        for i in range(n):
            dat = data[i]
            ax = plt.subplot(dim2, dim1, i + 1, frame_on=True)
            x, bins, p = ax.hist(dat[dat != 0], bins=bins, density=True)

            for item in p:
                item.set_height(item.get_height() / max(x))

            if (i % dim1 == 0):
                ax.set_ylabel(y_labels[int(i / dim1)])
            if (i < dim1):
                ax.set_title(x_labels[i])

            ax.set_ylim([0.0, 1.2])
            ax.text(0.0, 1.15, titles[i], **text_properties, **subtitle_font)

        plt.suptitle(plot_title, **title_font)
        plt.savefig(filename) if save else plt.show()

    def grid_images(X, plot_title, titles, x_labels, y_labels, save, filename):
        """
        Plot a grid of 2D map images.
        """
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        subtitle_font = {'weight': 'normal', 'size': 12, 'color': 'white'}
        title_font = {'weight': 'bold', 'size': 14}

        n = X.shape[0]
        dim1 = math.ceil(math.sqrt(n))
        dim2 = math.ceil(n / dim1)

        plt.figure()

        for i in range(n):
            image = np.matrix(X[i])
            ax = plt.subplot(dim2, dim1, i + 1, frame_on=False)
            ax.text(0.1, 0.1, titles[i], verticalalignment='top', horizontalalignment='left', **subtitle_font)

            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
            
            if (i % dim1 == 0):
                ax.set_ylabel(y_labels[int(i / dim1)])
            if (i < dim1):
                ax.set_title(x_labels[i])
            ax.imshow(image)

        plt.subplots_adjust(wspace=0.01, hspace=-0.19)
        plt.suptitle(plot_title, **title_font, y=1.0)
        plt.savefig(filename) if save else plt.show()

    def plot_galaxy(X, y, index):
        """
        Function to plot galaxies from 2D map data
        """
        galaxy_choice = np.matrix(X[index])
        plt.imshow(galaxy_choice, interpolation="nearest")
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
        x_eq = np.arange(0, 10, 1)
        y_eq = np.arange(0, 10, 1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(true, pred, 'ro', alpha=0.01)
        plt.plot(x_eq, y_eq, color='black', dashes=[2, 2])
        plt.title("2D Density")
        plt.xlabel(r"$P'_{rps}$")
        plt.ylabel(r'$P_{rps}$')
        # plt.show()
