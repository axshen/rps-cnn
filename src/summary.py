"""
Sample code for prediction of RPS parameters from inference with
pre-trained CNN model.
"""

from utils.rps_predictor import RPSPredictor


if __name__ == "__main__":
    # load model
    rps_predictor = RPSPredictor()
    rps_predictor.load("models/density-rho.h5")

    # print architecture summary
    rps_predictor.architecture()
