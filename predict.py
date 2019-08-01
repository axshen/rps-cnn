import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np

import gc
import glob
import sys
import matplotlib.pyplot as plt

import utils


class RPSPredictor():
    def __init__(self, n_maps, n_variables):
        self.model = None
        self.n_maps = n_maps
        self.n_variables = n_variables
        self.input_shape = (50, 50, n_maps)
    
    def load_model(self, path):
        """
        Loads a model from a specified path that contains the model
        and weights files (.h5 format).
        """
        try:
            self.model = load_model(path + "model.h5")
        except:
            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size = (3, 3),
                            activation = 'relu',
                            input_shape = input_shape))
            self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
            self.model.add(MaxPooling2D(pool_size = (2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation = 'relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(n_variables, activation = 'linear'))
            self.model.load_weights(path + "weights.h5")

    def predict(self, X):
        """
        Runs inference through the model that has been loaded. 
        """
        return self.model.predict(X)


def main():
    # user 
    test_num = '7'
    pred_var = 'r'
    map_type = 'density'
    nmodel = 9
    data_files = glob.glob("/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/data/additional/v2/*")
    model_path = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/archive/trained_models/m9.dir_e300_%s_%s/" % (map_type, pred_var)

    # variables determined from user properties
    n_variables = 2 if (pred_var == 'rv') or (pred_var == 'P_RPS') else 1
    n_maps = 2 if map_type == 'joint' else 1

    # read test data and performing conversions
    X_array = []
    titles = [
        "2d density, rho=0.15 * rho_halo",
        "2d kinematics, rho=0.15 * rho_halo",
        "2d density, rho=0.015",
        "2d kinematics, rho=0.015",
        "2d density, rho=0.045",
        "2d kinematics, rho=0.045"
    ]
    for file in data_files:
        X = utils.inout.read_file(file, nmodel, 1)
        X_array.append(X)
    X_array = np.array(X_array)

    # load model and perform inference
    rps_predictor = RPSPredictor(n_maps=n_maps, n_variables=n_variables)
    rps_predictor.load_model(model_path)

    for idx in range(X_array.shape[0]):
        images = X_array[idx]
        print(titles[idx])
        for i in range(1, nmodel):
            map2d = np.expand_dims(images[i], axis=0)
            y_pred = rps_predictor.predict(map2d)
            print(y_pred)
        print("\n")

    """
    # return mse
    print("Results and figures")
    test_error = utils.conversion.mse(y_pred, y_test)
    print('mse: %.5f' % test_error)
    plot_histogram(preds, pred_var)
    plot_compare(preds, y_test)
    """


if __name__ == "__main__":
    main()
