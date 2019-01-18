# -----------------------------------------------------------------------------------
# check v2 distribution after conversion

# -----------------------------------------------------------------------------------
# libraries

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# functions

# reading files and producing single output
def read_y_single(path,nmodel):
    # open file
    with open(path) as f:
        lines=f.readlines()
	# Y output
    ibin=0
    y=np.zeros((nmodel,1))
    for num,j in enumerate(lines[1:]):
        tm=j.strip().split()
        y[ibin,0]=float(tm[0])
        ibin+=1
    return(y)


# reading files and producing double output
def read_y_double(path,nmodel):
    # open file
    with open(path) as f:
        lines=f.readlines()
	# Y output
    ibin=0
    y=np.zeros((nmodel,2))
    for num,j in enumerate(lines[1:]):
        tm=j.strip().split()
        y[ibin,0]=float(tm[0])
        y[ibin,1]=float(tm[1])
        ibin+=1
    return(y)


# convert v to v^2 (normalised - proxy)
def v_to_v2_proxy(v):
    num = 10*((v-0.1)**2-(0-0.1)**2)
    denom = ((10-0.1)**2-(-0.1)**2)
    v2 = num/denom
    return(v2)


# convert v to v^2 (updated conversion)
def v_to_v2(v):
	v0 = v*(0.7-0.3)/(10)+0.3
	v0_min2 = 0.3**2
	v0_max2 = 0.7**2
	v2 = 10*(v0**2-v0_min2)/(v0_max2-v0_min2)
	return(v2)


# defining RPS pressure as potential prediction variable
def RPS_P(v,rho):
	rho0 = rho*(1-0.1)/(10)+0.1
	v0 = v*(0.7-0.3)/(10)+0.3
	P0 = rho0*v0**2
	P0_min = np.min(P0)
	P0_max = np.max(P0)
	P = 10*(P0-P0_min)/(P0_max-P0_min)
	return(P)


# plot comparison
def plot_single(x):
    plt.hist(x,color='green')
    plt.xlabel('Variable')
    plt.tight_layout()
    plt.show()


# plot comparison
def plot_compare(v,v2):
    plt.subplot(1,2,1)
    plt.hist(v)
    plt.xlabel('vrel')
    plt.subplot(1,2,2)
    plt.hist(v2,color='green')
    plt.xlabel('vrel2')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------

def main():
    nmodel=90000
    # vrel = read_y_single('../data/m1.dir_9_density/2dfvn.dat',nmodel)
    # vrel2 = v_to_v2(vrel)
    # plot_compare(vrel,vrel2)
    y = read_y_double('../data/m1.dir_9_density/2dfvn.dat',nmodel)
    P = RPS_P(y[:,0],y[:,1])
    plot_single(P)


# main()

# -----------------------------------------------------------------------------------
