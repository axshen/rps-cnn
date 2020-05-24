#!/usr/bin/env python3

import os
from utils.io import read_images, read_annots
from utils.metrics import rmse
from utils.rps_predictor import RPSPredictor


def main():
    # env
    root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

    # load test data
    test_dir = 'data/density/dir8/'
    n_test = int(1e4)
    X_t = read_images(os.path.join(root_dir, test_dir, '2dfv.dat'), n_test)
    y_t = read_annots(os.path.join(root_dir, test_dir, '2dfvn.dat'), n_test)
    print('finished reading test images')

    # load trained model
    model_dir = os.path.join(root_dir, 'code/models/', 'density-rho.h5')
    rps_model = RPSPredictor()
    rps_model.load(model_dir)
    print('trained model loading complete')

    # predict and evaluate
    print('running inference with trained model...')
    y_p = rps_model.predict(X_t)
    print(y_t.shape)
    print(y_p.shape)
    print('results:')
    print(rmse(y_t, y_p))


if __name__ == "__main__":
    main()
