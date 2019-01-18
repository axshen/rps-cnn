"""

This script takes a sample from the 2dfv.dat file (containing images of galaxies) and plots them. Galaxy images are 50x50
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

# argument parser from command line
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required=True, help='number of epochs')
args = vars(ap.parse_args())
epochs = int(args['epochs'])

# Choosing parameters
batch_size = 32
print('epochs = %s, batch size = %s' % (epochs, batch_size))

# path to training data files
path_den_train = '../data/m1.dir_8_density/'
path_kin_train = '../data/m1.dir_8_kinematics/'

# path to test data files
path_den_test = '../data/m1.dir_7_density/'
path_kin_test = '../data/m1.dir_7_kinematics/'

# parameters
num_classes = 2
n_mesh=50
nmodel=90000
img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh
input_shape = (img_rows, img_cols, 2)

# -----------------------------------------------------------------------
# function

# read function for map data
def read_data(path):

	# update to terminal
	print('reading data from: ' + str(path))

	# read files
	with open(path+'2dfv.dat') as f:
		lines=f.readlines()
	with open(path+'2dfvn.dat') as f:
		lines1=f.readlines()
	X=np.zeros((nmodel,n_mesh3))
	y=np.zeros((nmodel,2))

	# For 2D density map data
	ibin=0
	jbin=-1
	for num,j in enumerate(lines):
		jbin=jbin+1
		tm=j.strip().split()
		X[ibin,jbin]=float(tm[0])
		if jbin == n_mesh2:
			ibin+=1
			jbin=-1

	# Y output
	ibin=0
	for num,j in enumerate(lines1[1:]):
		tm=j.strip().split()
		y[ibin,0]=float(tm[0])
		y[ibin,1]=float(tm[1])
		ibin+=1

	# reshape
	X = X.reshape(X.shape[0], img_rows, img_cols, 1)

	# return data
	return(X, y)

# -----------------------------------------------------------------------
# reading training data

# extract training data
print('reading training data')
(train_X_den, train_y_den) = read_data(path_den_train)
(train_X_kin, train_y_kin) = read_data(path_kin_train)

# extracting test data
print('reading test data')
(test_X_den, test_y_den) = read_data(path_den_test)
(test_X_kin, test_y_kin) = read_data(path_kin_test)

# check equality in y train and test arrays and stop
train_equal = np.array_equal(train_y_den, train_y_kin)
test_equal = np.array_equal(test_y_den, test_y_kin)
print('train y array equal: ' + str(train_equal))
print('test y array equal: ' + str(train_equal))
if (train_equal == False | test_equal == False):
	sys.exit()
else:
	pass
y_train = train_y_den
y_test = test_y_den

# reconstruct x_train and x_test (join images as channels)
print('reconstructing training data')
x_train = np.concatenate((train_X_den, train_X_kin), axis=3)
x_test = np.concatenate((test_X_den, test_X_kin), axis=3)

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

# weights and models
model.save_weights("./output/m1.dir_9_e300_weights.h5")
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
