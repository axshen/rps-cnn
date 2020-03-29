#!/usr/bin/env python3 
import cv2
import numpy as np
import os


from utils.rps_predictor import RPSPredictor


if __name__ == "__main__":
    # load image
    img = cv2.imread("../sample_data/HI-dorado.jpg")
    img = np.expand_dims(img, axis=0)

    # load model
    rps_predictor = RPSPredictor()
    rps_predictor.load("../models/density-rho.h5")

    # inference
    y_p = rps_predictor.predict(img)
    print(y_p)
