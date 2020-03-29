""" Sample code for prediction of RPS parameters from inference with
pre-trained CNN model.
"""
import cv2
import numpy as np
import os


from utils.io import read_image
from utils.rps_predictor import RPSPredictor


if __name__ == "__main__":
    # load image
    img = read_image("sample_data/HI-dorado.jpg")

    # load model
    rps_predictor = RPSPredictor()
    rps_predictor.load("models/density-rho.h5")

    # inference
    y_p = rps_predictor.predict(img)
    print(f"Predicted rho_igm value: {y_p}")
