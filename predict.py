import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
from keras.applications import imagenet_utils
from keras.models import load_model
import keras.callbacks
import numpy as np

import gc
import os.path
import sys
import argparse
import random
import matplotlib.pyplot as plt

from utils import *

def main():

    # test file
    test_num = '8'
    pred_var = 'P_RPS'
    map_type = 'density'

    # Choosing parameters
    BATCH_SIZE = 64
    EPOCHS = 50
    NUM_CLASSES = 2 if (pred_var == 'rv') or (pred_var == 'P_RPS') else 1
    NUM_MAPS = 2 if map_type == 'joint' else 1
    nmodel = 80000 if test_num == '9' else 10000

    # parameters
    path_data = "../data/m1.dir_%s_%s/" % (test_num, map_type)
    path_model = "../gcloud_trained/m9.dir_e300_%s_%s/" % (map_type, pred_var)
    n_mesh = 50
    img_rows, img_cols = n_mesh, n_mesh
    n_mesh2 = n_mesh * n_mesh - 1
    n_mesh3 = n_mesh * n_mesh
    input_shape = (img_rows, img_cols, NUM_MAPS)

    # read test data and performing conversions
    print("Reading data")
    (x_test, y_test) = read_data(path_data, nmodel, NUM_CLASSES, pred_var)
    y_test = RPS_P(y_test[:,0], y_test[:,1])

    # sometimes model doesn't load from model.h5
    print("Loading model")
    try:
        model = load_model(path_model + "model.h5")
    except:
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (3, 3),
                         activation = 'relu',
                         input_shape = input_shape))
        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation = 'linear'))
        model.load_weights(path_model + "weights.h5")
        print(model.summary())
        print("Weights loaded")

    # inference
    model.compile(loss = 'mean_squared_error',
                  optimizer = keras.optimizers.Adadelta(),
    			  metrics = ['mse'])
    preds = model.predict(x_test, batch_size = BATCH_SIZE, verbose = 0, steps = None)
    gc.collect()

    # results
    print("Results and figures")
    test_error = mse(preds, y_test)
    print('mse: %.5f' % test_error)
    # plot_histogram(preds, pred_var)
    # plot_compare(preds, y_test)
    print("Plots completed and saved")


if __name__ == "__main__":
    main()
