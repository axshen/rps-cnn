import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
import utils

import keras
from keras.models import load_model

def main():
    path = "../../alternate_architectures_gcp/iter3/"
    history_data = "training_history.csv"
    model = "model.h5"
    test_data = "../../data/m1.dir_7_density/"

    # read training history    
    history = []
    with open(path + history_data, 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        next(reader)
        for line in reader:
            epoch = int(line[0])
            train_mse = float(line[1])
            val_mse = float(line[2])
            history.append([epoch, train_mse, val_mse])

    # plot
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], color = 'green', label = "train")
    plt.plot(history[:, 0], history[:, 2], color = 'blue', label = "val")
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(loc = 'upper right')
    plt.show()

    # evaluation against test
    (x_test, y_test) = utils.inout.read_data(test_data, 10000, 1, 'r')
    model = load_model(path + model)
    preds = model.predict(x_test, batch_size = 64, verbose = 0, steps = None)
    test_error = utils.conversion.mse(preds, y_test)
    print("Test error: %.4f" % test_error)

if __name__ == "__main__":
    main()
