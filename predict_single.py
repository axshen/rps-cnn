"""

Testing pre-saved weights on test data
By Austin Shen
17/08/2018

"""

# -----------------------------------------------------------------------
# libraries

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
from keras.applications import imagenet_utils
from keras.models import load_model
import keras.callbacks
import numpy as np

# system
import os.path
import sys
import argparse

# plotting packages
import random
import matplotlib.pyplot as plt

# written packages
import y_conversion as conversion

# -----------------------------------------------------------------------
# constants

print("Prediction with pre-trained weights")

# test file
test_num = '8'
pred_var = 'P_RPS'

# setting up paths
path_test_den = '../data/m1.dir_'+test_num+'_density/'
# path_test_kin = '../data/m1.dir_'+test_num+'_kinematics/'
path = '../gcloud_trained/m9.dir_e300_density_'+pred_var+'/'

# Choosing parameters
batch_size = 64
epochs = 50
num_classes = 2

# parameters
nmodel = 10000
nmodel_test = 10000
n_mesh=50
img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh

# need to change the input shape for joint vs den/vrel maps
input_shape = (img_rows, img_cols, 1)

# -----------------------------------------------------------------------
# read test data function (for joint)

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
	y=np.zeros((nmodel,num_classes))

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
		### Need to choose variable of interest ###
		### tm[0] <- vrel, tm[1] <- rho ###
		if (num_classes == 1):
			y[ibin,0]=float(tm[0])
		elif (num_classes == 2):
			y[ibin,0]=float(tm[0])
			y[ibin,1]=float(tm[1])
		ibin+=1

	# reshape
	X = X.reshape(X.shape[0], img_rows, img_cols, 1)

	# return data
	return(X, y)

# -----------------------------------------------------------------------
# read test data
print("Reading test data")

# read test data (for joint outputs)
(x_test, y_test) = read_data(path_test_den)
# (x_test, y_test) = read_data(path_test_kin)

# manipulation of test variable
y_test = conversion.RPS_P(y_test[:,0],y_test[:,1])

# -----------------------------------------------------------------------
# loading model from pre-trained

print("Reading model weights")

# read weights
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
model.add(Dense(1, activation='linear'))
model.load_weights(path + "m1.dir_9_e300_"+pred_var+"_weights.h5")

# model = load_model(path+'m9.dir_e300_'+pred_var+'_model.h5')
print(model.summary())
print("Weights loaded")

# model predict
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
			  metrics=['mse'])
preds = model.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

# -----------------------------------------------------------------------
# plot of data distribution (true test values)

# predicted and true values
pred = preds
true = y_test

# writing predictions to text file
f = open(path+'predictions_dir'+test_num+'.dat', 'w')
for i in range(len(pred)):
	f.write(str(pred[i][0])+'\n')
f.close()

# plotting histogram of values (for vrel and rho)
plt.hist(true)
plt.title("Histogram "+str(pred_var)+" (true)")
plt.xlabel(pred_var)
plt.ylabel('Frequency')
plt.savefig(path+'hist_'+pred_var+'_tm'+test_num+'.png')
plt.close()

# -----------------------------------------------------------------------
# plot performance

print("Performance plotting")
x_eq = np.arange(0,10,1)
y_eq = np.arange(0,10,1)

# test variable performance
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(true, pred, 'ro', alpha=0.01)
plt.plot(x_eq, y_eq, color='black',dashes=[2, 2])
plt.title("2D Density")
plt.xlabel(r"$P'_{rps}$")
plt.ylabel(r'$P_{rps}$')
plt.savefig(path+str(pred_var)+'_performance_tm'+test_num+'.png')
plt.close()

print("Plots completed and saved")

# -----------------------------------------------------------------------
