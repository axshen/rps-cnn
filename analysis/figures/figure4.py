import matplotlib.pyplot as plt
import numpy as np
import sys

import glob

sys.path.append("../../")

import utils


def main():
    path = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/data/additional/v1/"

    n = 9
    files = glob.glob(path + "*")
    files.sort()
    data = []
    for file in files:
        print(file)
        dat = utils.inout.read_file(file, n, 1)
        data.append(dat)

    data_array = np.array(data)
    data_array = np.array([
        data_array[0, 1, :, :, :],
        data_array[1, 4, :, :, :],
        data_array[0, 4, :, :, :],
        data_array[3, 1, :, :, :],
        data_array[4, 4, :, :, :],
        data_array[3, 4, :, :, :],
    ])

    write_file = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/plots/paper/FIG4.pdf"
    plot_title = ""
    titles = ["No RP", r"$\rho{}_{igm}=0.015$", r"$\rho{}_{igm}=0.15$", "", "", ""]
    plot_labels = None
    x_labels = ["T=0.0 Gyr", "T=0.14 Gyr", "T=0.14 Gyr"]
    y_labels = ["2D Density", "2D Kinematics", ""]

    data_array = data_array.reshape(6, 2500)
    utils.visualisation.grid_histogram(
        data_array, plot_title=plot_title, titles=titles, plot_labels=plot_labels, x_labels=x_labels, y_labels=y_labels, 
        wspace=0.0, hspace=0.0, 
        save=True, filename=write_file)


if __name__ == "__main__":
    main()
