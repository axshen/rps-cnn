import keras
import keras.callbacks
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils

import csv
import os.path
import sys
import argparse
import numpy as np
import random

import utils

def main():

    # changable parameters
    BATCH_SIZE = 32
    pred_var = "r"
    test_num = "8"
    map_type = "density"
    data_path = "../data/"

    # more constants
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", required = True, help = "number of epochs")
    args = vars(ap.parse_args())

    EPOCHS = int(args["epochs"])
    NUM_CLASSES = 2 if (pred_var == "rv") or (pred_var == "P_RPS") else 1
    NUM_MAPS = 2 if map_type == "joint" else 1

    path_train = data_path + "m1.dir_9_" + map_type + "/"
    path_test = data_path + "m1.dir_" + test_num + "_" + map_type + "/"

    # other constants
    n_mesh = 50
    nmodel_train = 90000
    nmodel_test = 10000
    img_rows, img_cols = n_mesh, n_mesh
    input_shape = (img_rows, img_cols, NUM_MAPS)

    # establishing model architecture
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
    model.compile(loss = 'mean_squared_error',
                  optimizer = keras.optimizers.Adadelta(),
    			  metrics = ['mse'])
    model.summary()

    sys.exit()

    # read test data and performing conversions
    print("Reading data")
    (x_train, y_train) = utils.inout.read_data(path_train, nmodel_train, NUM_CLASSES, pred_var)
    (x_test, y_test) = utils.inout.read_data(path_test, nmodel_test, NUM_CLASSES, pred_var)

    # conversions here if necessary


    # model training
    print("Training model")
    history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS,
              verbose = 1, validation_data = (x_test, y_test))

    # saving performance
    print("Saving model weights")
    model.save_weights("weights.h5")
    model.save("model.h5")

    # writing performance to file
    train_loss = list(history.history['loss'])
    val_loss = list(history.history['val_loss'])
    n_rows = len(train_loss)

    print("Writing training history to file")
    header = ['epoch', 'train_mse', 'val_mse']
    with open("training_history.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_rows):
            row = [i + 1, train_loss[i], val_loss[i]]
            writer.writerow(row)

if __name__ == "__main__":
    main()
