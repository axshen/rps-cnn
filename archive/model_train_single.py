"""

This script takes a sample from the 2dft.dat file (containing images of galaxies) and plots them. Galaxy images are 50x50
By Austin Shen
06/06/2018

"""

# -----------------------------------------------------------------------
# libraries

# tensorflow interface
import keras
import keras.callbacks
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.applications import imagenet_utils

# system
import os.path
import sys
import argparse

# other packages
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# constants

# Three files of interest:
# 2dft.dat - contains pixel values for images of galaxies (X)
# 2dftn1.dat - contains theta values corresponding to galaxy images (Y)
# 2dftn2.dat - contains phi values corresponding to galaxy images (Y)

'''
To execute this script you need to choose model parameters (see below)
The changable parameters include:
	- data used
	- batch size
	- number of epochs
'''

# argument parser from command line
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=True, help='number of epochs')
args = vars(ap.parse_args())
epochs = int(args['epochs'])

# path = '../data/m1.dir_6_density/'
path_train = str(raw_input('path to train data files: '))
path_test = str(raw_input('path to test data files: '))

# Choosing parameters
batch_size = 32
print('epochs = %s, batch size = %s' % (epochs, batch_size))

# parameters
num_classes = 2
n_mesh=50
nmodel=90000
img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh
input_shape = (img_rows, img_cols, 1)

# -----------------------------------------------------------------------
# read data

print("Reading training data")

# read files
with open(path_train+'2dfv.dat') as f:
  lines=f.readlines()
with open(path_train+'2dfvn.dat') as f:
  lines1=f.readlines()

x_train=np.zeros((nmodel,n_mesh3))
y_train=np.zeros((nmodel,2))

# For 2D density map data
ibin=0
jbin=-1
for num,j in enumerate(lines):
  jbin=jbin+1
  tm=j.strip().split()
  x_train[ibin,jbin]=float(tm[0])
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

# Y output
ibin=0
for num,j in enumerate(lines1[1:]):
  tm=j.strip().split()
  y_train[ibin,0]=float(tm[0])
  y_train[ibin,1]=float(tm[1])
  ibin+=1

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

# -----------------------------------------------------------------------
# set up training and test data

print("Reading test data")

# arrays for test data
x_test=np.zeros((nmodel,n_mesh3))
y_test=np.zeros((nmodel,2))

# read test data
with open(path_test+'2dfv.dat') as f:
  lines=f.readlines()
with open(path_test+'2dfvn.dat') as f:
  lines1=f.readlines()

# test X
ibin=0
jbin=-1
for num,j in enumerate(lines):
  jbin=jbin+1
  tm=j.strip().split()
  x_test[ibin,jbin]=float(tm[0])
  if jbin == n_mesh2:
    ibin+=1
    jbin=-1

# test Y
ibin=0
for num,j in enumerate(lines1[1:]):
  tm=j.strip().split()
  y_test[ibin,0]=float(tm[0])
  y_test[ibin,1]=float(tm[1])
  ibin+=1

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# -----------------------------------------------------------------------
# training models

print("Training model")

# Model Architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

# -----------------------------------------------------------------------
# save model weights

print("Saving model weights")

# serialize weights to HDF5
model.save_weights("./output/m1.dir_9_e300_weights.h5")

# save model to loadable file
model.save("./output/m1.dir_9_e300_model.h5")

# -----------------------------------------------------------------------
# plot results

print("Plotting performance")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./output/m1.dir_9_e300_accuracy.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./output/m1.dir_9_e300_loss.png')
plt.close()

# -----------------------------------------------------------------------
