#!/usr/local/bin/python3

import numpy as np
import sys
from utils import predictor, io, conversion, visualisation


if __name__ == "__main__":
    # ------ READING DATA -------
    # data
    val_image_path = "../data/density/dir7/2dfv.dat"
    val_annotation_path = "../data/density/dir7/2dfvn.dat"
    test_image_path = "../data/density/dir8/2dfv.dat"
    test_annotation_path = "../data/density/dir8/2dfvn.dat"

    # read test and validation data
    X_val = io.read_images(val_image_path)
    y_val = io.read_annotations(val_annotation_path)
    assert (X_val.shape[0] == y_val.shape[0]), "Number of images != annotations."
    nmodel = y_val.shape[0]
    y_val = y_val[:, 1]

    # read test and validation data
    X_test = io.read_images(test_image_path)
    y_test = io.read_annotations(test_annotation_path)
    assert (X_test.shape[0] == y_test.shape[0]), "Number of images != annotations."
    nmodel = y_test.shape[0]
    y_test = y_test[:, 1]
    
    # ------ LOAD MODELS ---------
    # selected model
    pred_var = 'r'
    map_type = 'density'
    pred_index = 0
    n_variables = 2 if (pred_var == 'rv') or (pred_var == 'P_RPS') else 1
    n_maps = 2 if map_type == 'joint' else 1

    # paths = ["./weights/alternate/iter%s/model.h5" % (x+1) for x in range(3)]
    # paths = ["./weights/m9.dir_e300_density_r/model.h5"]
    for model_path in paths:
        print("\n" + model_path)

        # load model and perform inference
        rps_predictor = predictor.RPSPredictor(
            n_maps=n_maps, n_variables=n_variables)
        rps_predictor.load_model(model_path)

        # validation performance
        yp_val = []
        for i in range(nmodel):
            map2d = np.expand_dims(X_val[i], axis=0)
            pred = rps_predictor.predict(map2d)
            yp_val.append(pred[0])
        yp_val = np.array(yp_val).reshape((nmodel))

        print("VALIDATION PERFORMANCE")
        val_error = conversion.rmse(yp_val, y_val)
        val_error_epsilon = conversion.mse(yp_val, y_val)
        print('rmse: %.5f' % val_error)
        print('mse: %.5f' % val_error_epsilon)

        # test performance
        yp_test = []
        for i in range(nmodel):
            map2d = np.expand_dims(X_test[i], axis=0)
            pred = rps_predictor.predict(map2d)
            yp_test.append(pred[0])
        yp_test = np.array(yp_test).reshape((nmodel))

        print("TEST PERFORMANCE")
        test_error = conversion.rmse(yp_test, y_test)
        test_error_epsilon = conversion.mse(yp_test, y_test)
        print('rmse: %.5f' % test_error)
        print('mse: %.5f' % test_error_epsilon)

