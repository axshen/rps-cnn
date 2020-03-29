#!/usr/local/bin/python3

import numpy as np
import sys
from utils import predictor, io, conversion, visualisation


if __name__ == "__main__":
    # selected model
    pred_var = 'r'
    map_type = 'density'
    pred_index = 1
    model_path = "./weights/m9.dir_e300_%s_%s/model.h5" % (map_type, pred_var)

    # data
    # image_path = "../data/additional/v3/m9.dir/2dfv.dat"
    # annotation_path = "../data/additional/v3/m9.dir/2dfvn.dat"
    # image_path_den = "../data/density/dir8/2dfv.dat"
    # image_path_kin = "../data/kinematic/dir8/2dfv.dat"
    #image_path = "../data/density/dir8/2dfv.dat"
    #annotation_path = "../data/density/dir8/2dfvn.dat"
    image_path = "../data/additional/v3/m11.dir/2dfv.dat"
    annotation_path = "../data/additional/v3/m11.dir/2dfvn.dat"

    # variables determined from user properties
    n_variables = 2 if (pred_var == 'rv') or (pred_var == 'P_RPS') else 1
    n_maps = 2 if map_type == 'joint' else 1

    # read test data
    # X_den = io.read_images(image_path_den)
    # X_kin = io.read_images(image_path_kin)
    X = io.read_images(image_path)
    y = io.read_annotations(annotation_path)
    assert (X.shape[0] == y.shape[0]), "Number of images != annotations."
    nmodel = y.shape[0]
    y = y[:, pred_index]

    # image conversions if necessary
    # X = np.concatenate((X_den, X_kin), axis=3)

    # conversions as necessary
    # y = y[:, pred_index].reshape((nmodel))
    # y = conversion.y_to_P_RPS(y)

    # load model and perform inference
    rps_predictor = predictor.RPSPredictor(
        n_maps=n_maps, n_variables=n_variables)
    rps_predictor.load_model(model_path)

    y_pred = []
    for i in range(nmodel):
        map2d = np.expand_dims(X[i], axis=0)
        pred = rps_predictor.predict(map2d)
        y_pred.append(pred[0])
    y_pred = np.array(y_pred).reshape((nmodel))

    # return mse
    print("Results and figures")
    test_error = conversion.rmse(y_pred, y)
    test_error_epsilon = conversion.mse(y_pred, y)
    print('rmse: %.5f' % test_error)
    print('mse: %.5f' % test_error_epsilon)
    visualisation.plot_compare(y_pred, y)