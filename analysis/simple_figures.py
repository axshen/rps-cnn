import matplotlib.pyplot as plt
import numpy as np
import sys

import glob
import keras
from keras.models import load_model

sys.path.append("..")

import utils


def main():
    path = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/data/send.dir/"
    model = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/archive/trained_models/m9.dir_e300_density_P_RPS/model.h5"

    n = 9
    files = glob.glob(path + "*")
    files.sort()
    data = []
    for file in files:
        print(file)
        dat = utils.inout.read_file(file, n, 1)
        data.append(dat)

    data_array = np.array(data)
    # data_array = np.array([
    #     data_array[0, 1, :, :, :],
    #     data_array[1, 4, :, :, :],
    #     data_array[0, 4, :, :, :],
    #     data_array[3, 1, :, :, :],
    #     data_array[4, 4, :, :, :],
    #     data_array[3, 4, :, :, :],
    # ])

    write_file = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/plots/paper/additional/FIG4.pdf"
    plot_title = "RP Histogram Difference"
    titles = ["No RP", r"$\rho{}_{igm}=0.015$", r"$\rho{}_{igm}=0.15$", "", "", ""]
    x_labels = ["T=0.0 Gyr", "T=0.14 Gyr", "T=0.14 Gyr"]
    y_labels = ["2D Density", "2D Kinematics", ""]

    utils.visualisation.grid_images(
        data_array, plot_title=plot_title, titles=titles, x_labels=x_labels, y_labels=y_labels, 
        save=False, filename=write_file)
    
    '''
    data_array = data_array.reshape(6, 2500)
    utils.visualisation.grid_histogram(
        data_array, plot_title=plot_title, titles=titles, x_labels=x_labels, y_labels=y_labels, 
        save=True, filename=write_file)
    '''

    '''
    data_path = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/data/m1.dir_8_density/"
    x_test, y_test = utils.inout.read_data(data_path, 10000, 1, "P_RPS")
    
    model = load_model(model)
    preds = model.predict(x_test, batch_size=64, verbose=0, steps=None)

    utils.visualisation.plot_compare(preds, y_test)
    plt.savefig("Fig7.pdf")
    '''


if __name__ == "__main__":
    main()
