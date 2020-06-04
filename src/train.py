#!/usr/bin/env python3

import os
from utils.io import read_images, read_annots
from utils.rps_predictor import RPSPredictor


def main():
    # env
    train_dir = os.getenv('TRAIN_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'data/density/dir9/'))
    val_dir = os.getenv('VAL_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'data/density/dir8/'))
    saved_model = os.getenv('MODEL_DIR', os.path.join(os.path.dirname(__file__), '../records/model'))

    # load data
    print("loading data...")
    n_train = int(9e4)
    X_train = read_images(os.path.join(train_dir, '2dfv.dat'), n_train)
    y_train = read_annots(os.path.join(train_dir, '2dfvn.dat'), n_train)
    print("finished reading training images")

    n_val = int(1e4)
    X_val = read_images(os.path.join(val_dir, '2dfv.dat'), n_val)
    y_val = read_annots(os.path.join(val_dir, '2dfvn.dat'), n_val)
    print("finished reading validation images")

    # initialise model
    print("initialising model...")
    rps_model = RPSPredictor()
    rps_model.compile()
    print("rps model initialised")

    # train
    print("starting model training...")
    rps_model.train(X_train, y_train, X_val, y_val, epochs=5)

    # save
    rps_model.save(saved_model)


if __name__ == "__main__":
    main()
