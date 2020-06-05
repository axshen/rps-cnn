#!/usr/bin/env python3

import os
import argparse
from utils.metrics import rmse
from utils.io import read_images, read_annots, split_annots
from utils.rps_predictor import RPSPredictor


def main():
    # env
    train_dir = os.getenv('TRAIN_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'data/density/dir9/'))
    val_dir = os.getenv('VAL_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'data/density/dir8/'))
    test_dir = os.getenv('TEST_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'data/density/dir7/'))
    saved_model = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(__file__), 'records/model'))

    # load trained model
    print("loading model...")
    rps_model = RPSPredictor()
    rps_model.load(saved_model)
    print("rps model initialised")

    # load data
    print("loading data...")
    # n_train = int(9e4)
    # X_train = read_images(os.path.join(train_dir, '2dfv.dat'), n_train)
    # y_train = read_annots(os.path.join(train_dir, '2dfvn.dat'), n_train)
    # _, rho_train = split_annots(y_train)
    # print("finished reading training images")

    n_val = int(1e4)
    X_val = read_images(os.path.join(val_dir, '2dfv.dat'), n_val)
    y_val = read_annots(os.path.join(val_dir, '2dfvn.dat'), n_val)
    _, rho_val = split_annots(y_val)
    print("finished reading validation images")

    n_test = int(1e4)
    X_test = read_images(os.path.join(test_dir, '2dfv.dat'), n_test)
    y_test = read_annots(os.path.join(test_dir, '2dfvn.dat'), n_test)
    _, rho_test = split_annots(y_test)
    print("finished reading validation images")

    # evaluate performance
    y_val_pred = rps_model.predict(X_val)
    print(y_val_pred.shape)
    print(rho_val.shape)
    val_rmse = rmse(rho_val, y_val_pred)
    print(val_rmse)

    y_test_pred = rps_model.predict(X_test)
    print(y_test_pred.shape)
    print(rho_test.shape)
    test_rmse = rmse(rho_test, y_test_pred)
    print(test_rmse)



if __name__ == "__main__":
    main()
