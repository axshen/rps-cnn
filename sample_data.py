"""

Takes large data file (90000 images, dir=m1.dir_4) and creates random samples for training and test to put in new directory (for training and test with different v_rel values). Input for the script in command line is number of random samples to take, and whether the file will be a training or test file (type).

By Austin Shen
20/08/2018

"""

# -----------------------------------------------------------------------
# libraries

import numpy as np

# system
import os.path
import sys
import argparse

# plotting packages
import random
import matplotlib
import matplotlib.pyplot as plt

# set seed for random
random.seed(101)

# -----------------------------------------------------------------------
# constants

# Three files of interest:
# 2dft.dat - contains pixel values for images of galaxies (X)
# 2dftn1.dat - contains theta values corresponding to galaxy images (Y)
# 2dftn2.dat - contains phi values corresponding to galaxy images (Y)

# argument parser from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--train", required=True,
				help='number training examples')
ap.add_argument("-o", "--test", required=True, help='number test examples')
args = vars(ap.parse_args())
train_samples = int(args['train'])
test_samples = int(args['test'])

# summary of images
path = 'm1.dir_4'
out_path = '../data/subsets/'
with open('../data/'+path+'/2dft.dat') as f:
    nmodel = int(sum(1 for _ in f)/(50*50))
print('nmodel =', nmodel)

num_classes = 2
n_mesh=50

img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh
input_shape = (img_rows, img_cols, 1)

# -----------------------------------------------------------------------
# read data

print("reading training data")

# read files
with open('../data/'+path+'/2dft.dat') as f:
  lines=f.readlines()
with open('../data/'+path+'/2dftn1.dat') as f:
  lines1=f.readlines()
with open('../data/'+path+'/2dftn2.dat') as f:
  lines2=f.readlines()

x_train=np.zeros((nmodel,n_mesh3))
y_train=np.zeros((nmodel,2))

# For 2D density map data
ibin=0
jbin=-1
for num,j in enumerate(lines):
  jbin=jbin+1
  tm=j.strip().split()
  x_train[ibin,jbin]=float(tm[0])
  # x_test[ibin,jbin]=float(tm[0])
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

# For morphological map (V)
ibin=0
for num,j in enumerate(lines1):
  tm=j.strip().split()
  y_train[ibin,0]=float(tm[0])
  ibin+=1

# For morphological map (rho)
ibin=0
for num,j in enumerate(lines2):
  tm=j.strip().split()
  y_train[ibin,1]=float(tm[0])
  ibin+=1

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
print(x_train.shape)
print(y_train.shape)

print('reading of training data complete')

# -----------------------------------------------------------------------
# random sampling of read data

print('writing files for output')

# output files
train_X_file = out_path+'train_X_n'+str(train_samples)+'.dat'
train_Y_file = out_path+'train_Y_n'+str(train_samples)+'.dat'
test_X_file = out_path+'test_X_n'+str(test_samples)+'.dat'
test_Y_file = out_path+'test_Y_n'+str(test_samples)+'.dat'

# choosing index
indexes = np.random.choice(range(nmodel),size=(train_samples+test_samples), replace=False)
train_indexes = indexes[0:train_samples]
test_indexes = indexes[train_samples:(train_samples+test_samples)]

# sample for training and test sets
train_X_sample = x_train[train_indexes,:,:,:]
train_Y_sample = y_train[train_indexes,:]
test_X_sample = x_train[test_indexes,:,:,:]
test_Y_sample = y_train[test_indexes,:]

# writing training data
print('writing training X data')
f = open(train_X_file, 'w')
train_X_sample_write = train_X_sample.flatten()
for i in range(0,len(train_X_sample_write),1):
	f.write(str(train_X_sample_write[i])+'\n')
f.close()
print('writing training Y data')
f = open(train_Y_file, 'w')
for i in range(0,train_Y_sample.shape[0],1):
	f.write(str(train_Y_sample[i,0])+'	'+str(train_Y_sample[i,1])+'\n')
f.close()

print('writing test X data')
f = open(test_X_file, 'w')
test_X_sample_write = test_X_sample.flatten()
for i in range(0,len(test_X_sample_write),1):
	f.write(str(test_X_sample_write[i])+'\n')
f.close()
print('writing test Y data')
f = open(test_Y_file, 'w')
for i in range(0,test_Y_sample.shape[0],1):
	f.write(str(test_Y_sample[i,0])+'	'+str(test_Y_sample[i,1])+'\n')
f.close()

print('writing complete')

# -----------------------------------------------------------------------
