#!/usr/local/bin/python3

import glob
import numpy as np
import sys
sys.path.append('../')

from utils import io, visualisation


if __name__ == "__main__":
    path = "../../data/additional/v1/"

    files = sorted(glob.glob(path + "*"))
    data = []
    for f in files:
        imgs = io.read_images(f)
        data.append(imgs)

    data_array = np.array(data)
    data_array = data_array[3]

    write_file = "../../plots/paper/FIG2.pdf"
    plot_title = ""
    titles = ["%.3f" % float(i * 0.035) for i in range(9)]
    titles[0] = r"$T=0.000$ Gyr"
    x_labels = None
    y_labels = None

    visualisation.grid_images(
        data_array, plot_title=plot_title, titles=titles,
        x_labels=x_labels, y_labels=y_labels,
        wspace=-0.5, hspace=-0.01,
        save=False, filename=write_file)
