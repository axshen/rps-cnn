#!/usr/local/bin/python3

import glob
import numpy as np
import sys
sys.path.append("../")

from utils import io, visualisation


if __name__ == "__main__":
    path = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/data/additional/v1/"

    n = 9
    files = glob.glob(path + "*")
    files.sort()
    data = []
    for f in files:
        X = io.read_images(f)
        data.append(X)

    data_array = np.array(data)
    data_array = np.array([
        data_array[0, 1, :, :, :],
        data_array[1, 4, :, :, :],
        data_array[0, 4, :, :, :],
        data_array[3, 1, :, :, :],
        data_array[4, 4, :, :, :],
        data_array[3, 4, :, :, :],
    ])

    write_file = "../../plots/paper/FIG3.pdf"
    plot_title = ""
    titles = [
        "No RP",
        r"$\rho{}_{\rm igm}=0.015\rho{}_{\rm dm}$",
        r"$\rho{}_{\rm igm}=0.15\rho{}_{\rm dm}$",
        "", "", ""]
    x_labels = ["T=0.0 Gyr", "T=0.14 Gyr", "T=0.14 Gyr"]
    y_labels = ["2D Density", "2D Kinematics", ""]

    visualisation.grid_images(
        data_array, plot_title=plot_title, titles=titles,
        x_labels=x_labels, y_labels=y_labels,
        wspace=0.0, hspace=-0.2,
        save=True, filename=write_file)
