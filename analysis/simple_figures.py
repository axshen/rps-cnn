import matplotlib.pyplot as plt
import numpy as np 
import sys

sys.path.append("..")
import utils

import keras
from keras.models import load_model

def main():
    data_path = "../../data/m1.dir_8_density/"
    model = "../../gcloud_trained/m9.dir_e300_density_P_RPS/model.h5"

    (x_test, y_test_both) = utils.inout.read_data(data_path, 10000, 2, 'both')
    y_test = utils.conversion.RPS_P(y_test_both[:, 0], y_test_both[:, 1])

    model = load_model(model)
    preds = model.predict(x_test, batch_size = 64, verbose = 0, steps = None)

    utils.visualisation.plot_compare(preds, y_test)
    plt.savefig("Fig7.eps", format = 'eps', dpi = 300)

if __name__ == "__main__":
    main()