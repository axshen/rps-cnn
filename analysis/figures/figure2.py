import matplotlib.pyplot as plt
import numpy as np
import sys

import glob

sys.path.append("../..")

import utils

if __name__ == "__main__":
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
    data_array = data_array[3]

    write_file = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/plots/paper/FIG2.pdf"
    plot_title = ""
    titles = ["%.3f Gyr" % float(i * 0.035) for i in range(9)]
    x_labels = None
    y_labels = None

    utils.visualisation.grid_images(
        data_array, plot_title=plot_title, titles=titles, x_labels=x_labels, y_labels=y_labels, 
        wspace=-0.5, hspace=-0.05,
        save=True, filename=write_file)
