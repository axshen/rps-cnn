import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

# ----------------------------------------------------------------------
# functions

def read_dat(file):
    with open(file) as f:
        lines=f.readlines()
    n = len(lines)
    preds = np.zeros((n,1))
    for i in range(n):
        preds[i] = float(lines[i])
    return(preds)

def read_y(path,nmodel,index):
    # open file
    with open(path) as f:
        lines=f.readlines()
    # Y output
    ibin=0
    y=np.zeros((nmodel,1))
    for num,j in enumerate(lines[1:]):
        tm=j.strip().split()
        y[ibin,0]=float(tm[index])
        ibin+=1
    return(y)

def overlap_predictions(dat_joined_1,dat_joined_2,dat_den_1,dat_den_2,dat_kin_1,dat_kin_2):
    bins = np.linspace(-10,10,50)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.subplot(1,2,1)
    plt.hist(dat_joined_1, bins, alpha=0.3, label='joined')
    plt.hist(dat_den_1, bins, alpha=0.3, label='density')
    plt.hist(dat_kin_1, bins, alpha=0.3, label='kinematics')
    plt.legend(loc='upper left')
    plt.title('Prediction comparison (m8)')
    plt.ylabel('Count')
    plt.xlabel('vrel (pred)')

    plt.subplot(1,2,2)
    plt.hist(dat_joined_2, bins, alpha=0.3, label='joined')
    plt.hist(dat_den_2, bins, alpha=0.3, label='density')
    plt.hist(dat_kin_2, bins, alpha=0.3, label='kinematics')
    plt.legend(loc='upper left')
    plt.title('Prediction comparison (m7)')
    plt.ylabel('Count')
    plt.xlabel('vrel (pred)')
    plt.tight_layout()
    plt.show()

def subplot_performance(dat_den,dat_kin,dat_both,truth):
    x_eq = np.arange(0,10,1)
    y_eq = np.arange(0,10,1)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax = [plt.subplot(3,1,i+1) for i in range(3)]
    #for a in ax:
    #    a.set_xticklabels([])
    #    a.set_yticklabels([])
    plt.subplots_adjust(wspace=0., hspace=0.)

    plt_min = -0.5
    plt_max_x = 10.3
    plt_max_y = 11.5

    plt.subplot(3,1,1)
    plt.plot(truth,dat_den,'.',color='blue',alpha=0.01)
    plt.plot(x_eq, y_eq, color='black',dashes=[2, 2])
    plt.ylim((plt_min,plt_max_y))
    plt.xlim((plt_min,plt_max_x))
    plt.text(0,10,'2D Density')

    plt.subplot(3,1,2)
    plt.plot(truth,dat_kin,'.',color='red',alpha=0.01)
    plt.plot(x_eq, y_eq, color='black',dashes=[2, 2])
    plt.ylim((plt_min,plt_max_y))
    plt.xlim((plt_min,plt_max_x))
    plt.text(0,10,'2D Kinematics')

    plt.subplot(3,1,3)
    plt.plot(truth,dat_both,'.',color='purple',alpha=0.01)
    plt.plot(x_eq, y_eq, color='black',dashes=[2, 2])
    plt.ylim((plt_min,plt_max_y))
    plt.xlim((plt_min,plt_max_x))
    plt.text(0,10,'2D Joined')

    ax[2].set_xlabel(r'$\rho{}_{igm}$',size='large')
    ax[1].set_ylabel(r"$\rho{}'_{igm}$",size='large')

    plt.show()

def subplot_histograms(dat_den,dat_kin,dat_both,truth):
    den_perf = np.absolute((dat_den-truth)/truth)
    kin_perf = np.absolute((dat_kin-truth)/truth)
    both_perf = np.absolute((dat_both-truth)/truth)
    bins = np.linspace(0,3,50)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax = [plt.subplot(3,1,i+1) for i in range(3)]
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
    plt.subplots_adjust(wspace=0., hspace=0.)

    plt_min = -0.1
    plt_max_x = 2
    plt_max_y = int(3.5e3)

    plt.subplot(3,1,1)
    plt.hist(den_perf, bins, alpha=0.7, color='blue')
    plt.ylim((plt_min,plt_max_y))
    plt.xlim((plt_min,plt_max_x))
    plt.text(1.5,int(3e3),'2D Density')

    plt.subplot(3,1,2)
    plt.hist(kin_perf, bins, alpha=0.7, color='red')
    plt.ylim((plt_min,plt_max_y))
    plt.xlim((plt_min,plt_max_x))
    plt.text(1.5,int(3e3),'2D Kinematics')

    plt.subplot(3,1,3)
    plt.hist(both_perf, bins, alpha=0.7, color='purple')
    plt.ylim((plt_min,plt_max_y))
    plt.xlim((plt_min,plt_max_x))
    plt.text(1.5,int(3e3),'2D Joined')

    ax[2].set_xlabel(r'$|\frac{v{}_{pred}-v_{c}}{v_{c}}|$',size='large')
    ax[1].set_ylabel("N",size='large')

    plt.show()

def main():

    # constants and directories
    root_dir = '../gcloud_trained/'
    dir1 = 'm9.dir_e300_joint_v/'
    dir2 = 'm9.dir_e300_density_v/'
    dir3 = 'm9.dir_e300_kinematics_v/'
    dir4 = 'm9.dir_e300_joint_r/'
    dir5 = 'm9.dir_e300_density_r/'
    dir6 = 'm9.dir_e300_kinematics_r/'
    f8 = 'predictions_dir8.dat'
    f7 = 'predictions_dir7.dat'

    # expected values
    truth_dir = '../data/'
    dir_8 = 'm1.dir_8_density/'
    dir_7 = 'm1.dir_7_density/'
    f_y = '2dfvn.dat'

    # reading prediction files
    dat_joined_1 = read_dat(root_dir+dir1+f8)
    dat_joined_2 = read_dat(root_dir+dir1+f7)
    dat_den_1 = read_dat(root_dir+dir2+f8)
    dat_den_2 = read_dat(root_dir+dir2+f7)
    dat_kin_1 = read_dat(root_dir+dir3+f8)
    dat_kin_2 = read_dat(root_dir+dir3+f7)

    # reading truth value
    rho_8 = read_y(truth_dir+dir_8+f_y,nmodel=10000,index=1)
    rho_7 = read_y(truth_dir+dir_7+f_y,nmodel=10000,index=1)

    # compare outputs
    overlap_predictions(dat_joined_1,dat_joined_2,dat_den_1,dat_den_2,dat_kin_1,dat_kin_2)

    # plotting performance plots
    subplot_performance(dat_den_1,dat_kin_1,dat_joined_1,rho_8)
    subplot_histograms(dat_den_1,dat_kin_1,dat_joined_1,rho_8)

    # calculations of mse for different input data
    mse_joined = np.mean((dat_joined_1 - rho_8)**2)
    mse_den = np.mean((dat_den_1 - rho_8)**2)
    mse_kin = np.mean((dat_kin_1 - rho_8)**2)
    print('mse: joined = %.5f, den = %.5f, kin = %.5f' % (mse_joined, mse_den, mse_kin))

    # evaluations of performance for velocity
    dat_joined_1_v = read_dat(root_dir+dir4+f8)
    dat_joined_2_v = read_dat(root_dir+dir4+f7)
    dat_den_1_v = read_dat(root_dir+dir5+f8)
    dat_den_2_v = read_dat(root_dir+dir5+f7)
    dat_kin_1_v = read_dat(root_dir+dir6+f8)
    dat_kin_2_v = read_dat(root_dir+dir6+f7)
    vrel_8 = read_y(truth_dir+dir_8+f_y,nmodel=10000,index=0)
    vrel_7 = read_y(truth_dir+dir_7+f_y,nmodel=10000,index=0)
    subplot_performance(dat_den_1_v,dat_kin_1_v,dat_joined_1_v,vrel_8)
    subplot_histograms(dat_den_1_v,dat_kin_1_v,dat_joined_1_v,vrel_8)
    mse_joined_v = np.mean((dat_joined_1_v - vrel_8)**2)
    mse_den_v = np.mean((dat_den_1_v - vrel_8)**2)
    mse_kin_v = np.mean((dat_kin_1_v - vrel_8)**2)
    print('mse: joined = %.5f, den = %.5f, kin = %.5f' % (mse_joined_v, mse_den_v, mse_kin_v))

main()

