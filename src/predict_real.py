#!/usr/local/bin/python3

import numpy as np
from utils import predictor, io


if __name__ == "__main__":
    # selected variables
    pred_var = 'r'
    map_type = 'density'
    image_file = "../data/rps_real.jpg"
    model_path = "./weights/m9.dir_e300_%s_%s/model.h5" % (map_type, pred_var)

    # variables determined from user properties
    n_variables = 2 if (pred_var == 'rv') or (pred_var == 'P_RPS') else 1
    n_maps = 2 if map_type == 'joint' else 1

    # read test data
    X = io.read_image(image_file)

    # load model
    rps_predictor = predictor.RPSPredictor(
        n_maps=n_maps, n_variables=n_variables)
    rps_predictor.load_model(model_path)

    # prediction
    map2d = np.expand_dims(X, axis=0)
    y_pred = rps_predictor.predict(map2d)
    print(y_pred)
