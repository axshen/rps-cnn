#!/usr/bin/env python3

import os
from utils.io import read_images, read_annots
from utils.rps_predictor import RPSPredictor


def main():
    # env
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

    # load data
    train_dir = 'data/density/dir9/'
    n_train = int(9e4)
    X_train = read_images(os.path.join(root_dir, train_dir, '2dfv.dat'), n_train)
    y_train = read_annots(os.path.join(root_dir, train_dir, '2dfvn.dat'), n_train)

    val_dir = 'data/density/dir8/'
    n_val = int(1e4)
    X_val = read_images(os.path.join(root_dir, val_dir, '2dfv.dat'), n_val)
    y_val = read_annots(os.path.join(root_dir, val_dir, '2dfvn.dat'), n_val)

    # initialise model
    rps_model = RPSPredictor()
    rps_model.compile()

    # train
    rps_model.train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
