import matplotlib.pyplot as plt
import numpy as np
import sys

import glob

sys.path.append("../..")

import utils

if __name__ == "__main__":
    path = "/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/data/additional/v1/"

    n = 9
    files = glob.glob(path + "*")
    files.sort()
    data = []
    for file in files:
        dat = utils.inout.read_file(file, n, 1)
        data.append(dat)

    data_array = np.array(data)
    data_array = data_array[0]
    img = data_array[-1,:,:,0]

    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("galaxy.png", transparent=True)
