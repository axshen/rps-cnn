import matplotlib.pyplot as plt
import numpy as np
import os
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

def read_y(path, nmodel, index):
    # open file
    with open(path) as f:
        lines = f.readlines()
    # Y output
    ibin = 0
    y = np.zeros((nmodel, 1))
    for num, j in enumerate(lines[1:]):
        tm = j.strip().split()
        y[ibin, 0]=float(tm[index])
        ibin += 1
    return(y)


def subplot_histograms(dat_den, dat_kin, dat_both, truth, save):
    den_perf = np.absolute((dat_den-truth) / truth)
    kin_perf = np.absolute((dat_kin-truth) / truth)
    both_perf = np.absolute((dat_both-truth) / truth)
    bins = np.linspace(0, 3, 100)

    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')

    ax = [plt.subplot(1, 3, i + 1) for i in range(3)]
    for a in ax:
        if a != ax[0]:
            a.set_yticklabels([])
    plt.subplots_adjust(wspace = 0.1, hspace = 0.)

    plt_min = 0.0
    plt_max_x = 1.8
    plt_max_y = 1.05

    plt.subplot(1, 3, 1)
    x, bins, p = plt.hist(den_perf, bins, alpha = 0.7, color='blue', density=True)
    for item in p:
        item.set_height(item.get_height() / max(x))
    plt.ylim((plt_min, plt_max_y))
    plt.xlim((plt_min, plt_max_x))
    plt.title('2D Density')

    plt.subplot(1, 3, 2)
    x, bins, p = plt.hist(kin_perf, bins, alpha = 0.7, color = 'red', density=True)
    for item in p:
        item.set_height(item.get_height() / max(x))
    plt.ylim((plt_min, plt_max_y))
    plt.xlim((plt_min, plt_max_x))
    plt.title('2D Kinematics')

    plt.subplot(1, 3, 3)
    x, bins, p = plt.hist(both_perf, bins, alpha = 0.7, color = 'purple', density=True)
    for item in p:
        item.set_height(item.get_height() / max(x))
    plt.ylim((plt_min, plt_max_y))
    plt.xlim((plt_min, plt_max_x))
    plt.title('2D Joined')

    # ax[0].set_xlabel(r'$|\frac{\rho{}_{igm,p} - \rho_{igm,c}}{\rho_{igm,c}}|$', fontsize=12)
    # ax[1].set_xlabel(r'$|\frac{\rho{}_{igm,p} - \rho_{igm,c}}{\rho_{igm,c}}|$', fontsize=12)
    # ax[2].set_xlabel(r'$|\frac{\rho{}_{igm,p} - \rho_{igm,c}}{\rho_{igm,c}}|$', fontsize=12)
    ax[0].set_xlabel(r'$\chi{}$', fontsize=12)
    ax[1].set_xlabel(r'$\chi{}$', fontsize=12)
    ax[2].set_xlabel(r'$\chi{}$', fontsize=12)
    ax[0].set_ylabel("N", fontsize=16)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    if save:
        plt.savefig('FIG6.pdf')
    else:
        plt.show()

def main():
    # constants and directories
    root_dir = '/Users/austin.shen/Dropbox/UWA/ICRAR_Kenji/archive/trained_models/'
    dir1 = 'm9.dir_e300_joint_v/'
    dir2 = 'm9.dir_e300_density_v/'
    dir3 = 'm9.dir_e300_kinematics_v/'
    dir4 = 'm9.dir_e300_joint_r/'
    dir5 = 'm9.dir_e300_density_r/'
    dir6 = 'm9.dir_e300_kinematics_r/'
    f8 = 'predictions_dir8.dat'
    f7 = 'predictions_dir7.dat'

    # expected values
    truth_dir = '../../../data/'
    dir_8 = 'm1.dir_8_density/'
    dir_7 = 'm1.dir_7_density/'
    f_y = '2dfvn.dat'

    # reading prediction files
    dat_joined_1 = read_dat(root_dir + dir4 + f8)
    dat_joined_2 = read_dat(root_dir + dir4 + f7)
    dat_den_1 = read_dat(root_dir + dir5 + f8)
    dat_den_2 = read_dat(root_dir + dir5 + f7)
    dat_kin_1 = read_dat(root_dir + dir6 + f8)
    dat_kin_2 = read_dat(root_dir + dir6 + f7)
    rho_8 = read_y(truth_dir + dir_8 + f_y, nmodel = 10000, index = 1)
    rho_7 = read_y(truth_dir + dir_7 + f_y, nmodel = 10000, index = 1)

    # plotting performance plots
    subplot_histograms(
        dat_den_1, dat_kin_1, dat_joined_1, rho_8, 
        save=True)


if __name__ == "__main__":
    main()

