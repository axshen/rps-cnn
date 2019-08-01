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
    with open(path) as f:
        lines = f.readlines()
    ibin = 0
    y = np.zeros((nmodel, 1))
    for num, j in enumerate(lines[1:]):
        tm = j.strip().split()
        y[ibin, 0]=float(tm[index])
        ibin += 1
    return(y)

def subplot_performance(dat_den, dat_kin, dat_both, truth, save):
    x_eq = np.arange(0, 10, 1)
    y_eq = np.arange(0, 10, 1)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax = [plt.subplot(3, 1, i + 1) for i in range(3)]
    plt.subplots_adjust(wspace=0., hspace=0.)

    plt_min = -0.5
    plt_max_x = 10.3
    plt_max_y = 11.5

    plt.subplot(3, 1, 1)
    plt.plot(truth, dat_den, '.', color='blue', alpha=0.01)
    plt.plot(x_eq, y_eq, color='black', dashes=[2, 2])
    plt.ylim((plt_min, plt_max_y))
    plt.xlim((plt_min, plt_max_x))
    plt.text(0, 10, '2D Density')

    plt.subplot(3, 1, 2)
    plt.plot(truth, dat_kin, '.', color='red', alpha=0.01)
    plt.plot(x_eq, y_eq, color='black', dashes=[2, 2])
    plt.ylim((plt_min, plt_max_y))
    plt.xlim((plt_min, plt_max_x))
    plt.text(0, 10, '2D Kinematics')

    plt.subplot(3, 1, 3)
    plt.plot(truth, dat_both, '.', color='purple', alpha=0.01)
    plt.plot(x_eq, y_eq, color='black', dashes=[2, 2])
    plt.ylim((plt_min, plt_max_y))
    plt.xlim((plt_min, plt_max_x))
    plt.text(0, 10, '2D Joined')

    ax[2].set_xlabel(r'$\rho{}_{igm,c}$', fontsize=16)
    ax[1].set_ylabel(r"$\rho{}_{igm,p}$", fontsize=16)

    # plt.tight_layout()
    if save:
        plt.savefig('FIG5.pdf')
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
    subplot_performance(
        dat_den_1, dat_kin_1, dat_joined_1, rho_8, 
        save=True)


if __name__ == "__main__":
    main()

