import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import binned_statistic
import sys
import glob

def read_dat(file):
    with open(file) as f:
        lines = f.readlines()
    n = len(lines)
    preds = np.zeros((n, 1))
    for i in range(n):
        preds[i] = float(lines[i])
    return(preds)

def read_P_RPS(path, nmodel):
    with open(path) as f:
        lines = f.readlines()
    ibin = 0
    y = np.zeros((nmodel, 2))
    for num, j in enumerate(lines[1:]):
        tm = j.strip().split()
        v = float(tm[0])
        r = float(tm[1])
        y[ibin, 0] = v
        y[ibin, 1] = r
        ibin += 1
    return(y)

def P_RPS(v, rho):
    """
    Defining RPS pressure as potential prediction variable
    """
    rho0 = rho * (1 - 0.1) / (10) + 0.1
    v0 = v * (0.7 - 0.3) / (10) + 0.3
    P0 = rho0 * v0 ** 2
    P0_min = min(P0)
    P0_max = max(P0)
    # P0_min = 0.009
    # P0_max = 0.49
    P = 10.0 * (P0 - P0_min) / (P0_max - P0_min)
    return P

def main():
    # constants and directories
    root_dir = '/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/archive/trained_models/'
    preds_dir = 'm9.dir_e300_density_P_RPS/'

    f8 = 'predictions_dir8.dat'
    f7 = 'predictions_dir7.dat'

    # expected values
    truth_dir = '../../../data/'
    dir_8 = 'm1.dir_8_density/'
    dir_7 = 'm1.dir_7_density/'
    f_y = '2dfvn.dat'

    # reading prediction files
    preds_8 = read_dat(root_dir + preds_dir + f8)
    preds_7 = read_dat(root_dir + preds_dir + f7)
    y_8 = read_P_RPS(truth_dir + dir_8 + f_y, nmodel = 10000)
    y_7 = read_P_RPS(truth_dir + dir_7 + f_y, nmodel = 10000)

    P_RPS_8 = P_RPS(y_8[:,0], y_8[:,1])

    # plotting
    x = np.linspace(0, 10, 500)
    y = np.linspace(0, 10, 500)

    # binned error data
    bin_x = np.array([0.5 + i for i in range(10)])
    bin_means = binned_statistic(np.array(P_RPS_8), np.array(preds_8).reshape(len(preds_8)), statistic="mean", bins=10)[0]
    bin_std = binned_statistic(np.array(P_RPS_8), np.array(preds_8).reshape(len(preds_8)), statistic="std", bins=10)[0]
    # bin_counts = binned_statistic(np.array(P_RPS_8), np.array(preds_8).reshape(len(preds_8)), statistic="count", bins=10)[0]
    # bin_error = bin_std / np.sqrt(bin_counts)

    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.plot(x, y, dashes=[6, 2], color='black', alpha=0.5)
    plt.scatter(P_RPS_8, preds_8, marker='.', color='red', alpha=0.1)
    (_, caps, _) = plt.errorbar(bin_x, bin_means, yerr=bin_std, color='black', alpha=0.5, fmt='o', markersize=2, capsize=5)
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.xlabel(r"$P_{rps,c}$", fontsize=14)
    plt.ylabel(r"$P_{rps,p}$", fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../../../plots/paper/FIG7.pdf")


if __name__ == "__main__":
    main()

