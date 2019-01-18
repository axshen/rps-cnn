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

# -----------------------------------------------------------------------
# constants

print("Prediction with pre-trained weights")

# setting up paths
path_test_den = '../data/m1.dir_8_density/'
path_test_kin = '../data/m1.dir_8_kinematics/'
path = '../gcloud_trained/m9.dir_e300_joint_r/'

# Choosing parameters
batch_size = 64
epochs = 50
num_classes = 1

# parameters
nmodel = 10000
nmodel_test = 10000
n_mesh=50
img_rows, img_cols = n_mesh, n_mesh
n_mesh2=n_mesh*n_mesh-1
n_mesh3=n_mesh*n_mesh
input_shape = (img_rows, img_cols, 2)

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
(test_X_den, test_y_den) = read_data(path_test_den)
(test_X_kin, test_y_kin) = read_data(path_test_kin)
y_test = test_y_den
x_test = np.concatenate((test_X_den, test_X_kin), axis=3)

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
model.add(Dense(num_classes, activation='linear'))
model.load_weights(path + "m1.dir_9_e300_joint_rho_weights.h5")
# model = load_model(path+'m1.dir_9_e300_joint_rho_model.h5')
print(model.summary())
print("Weights loaded")

# model predict
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])
preds = model.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

# -----------------------------------------------------------------------
# plot of data distribution (true test values)

# writing predictions of single values
f = open(path+'predictions_dir8.dat', 'w')
for i in range(len(preds)):
	f.write(str(preds[i][0])+'\n')
f.close()

sys.exit()

'''
# writing predictions to text file
f = open(path+'predictions_dir7.dat', 'w')
for i in range(len(preds)):
	f.write(str(preds[i][0])+','+preds[i][1]+'\n')
f.close()
'''

# predicted values
vrel_pred = preds[:,0]
rho_pred = preds[:,1]
print("V_rel (pred):", vrel_pred)
print("rho (pred):", rho_pred)

# true values
vrel_true = y_test[:,0]
rho_true = y_test[:,1]
print("V_rel (true):", vrel_true)
print("rho (true):", rho_true)

# plotting histogram of values (for vrel and rho)
plt.hist(vrel_true)
plt.title("Histogram of V_rel (true)")
plt.xlabel("V_rel")
plt.ylabel('Frequency')
plt.savefig(path+'hist_vrel_tm7.png')
plt.close()

plt.hist(rho_true)
plt.title('Histogram of rho (true)')
plt.xlabel('Density (rho)')
plt.ylabel('Frequency')
plt.savefig(path+'hist_rho_tm7.png')
plt.close()

# -----------------------------------------------------------------------
# plot performance

print("Performance plotting")
x_eq = np.arange(0,10,1)
y_eq = np.arange(0,10,1)

# vrel performance
plt.plot(vrel_true, vrel_pred, 'ro', alpha=0.02)
plt.plot(x_eq, y_eq, color='blue')
plt.title("Performance (V_rel)")
plt.xlabel("V_rel (true)")
plt.ylabel('V_rel (pred)')
plt.savefig(path+'vrel_performance_tm7.png')
plt.close()

# density performance
plt.plot(rho_true, rho_pred, 'ro', alpha=0.02)
plt.plot(x_eq, y_eq, color='blue')
plt.title("Performance (rho)")
plt.xlabel("rho (true)")
plt.ylabel('rho (pred)')
plt.savefig(path+'rho_performance_tm7.png')
plt.close()

# calculating cosine distance
cosine_distance = np.zeros((nmodel_test))
for i in range(0,nmodel_test,1):
	dp = np.dot(preds[i,:], y_test[i,:])
	denom = np.linalg.norm(preds[i,:])*np.linalg.norm(y_test[i,:])
	cosine_distance[i] = dp/denom
print('max cosine distance: %s' % max(cosine_distance))
print('min cosine distance: %s' % min(cosine_distance))
# cosine similarity histogram
plt.hist(cosine_distance[np.logical_not(np.isnan(cosine_distance))])
plt.axvline(x=np.mean(cosine_distance), color='red')
plt.title("Cosine Distance Histogram")
plt.xlabel("Sample")
plt.ylabel("Cosine Distance")
plt.savefig(path+'cosine_histogram_tm7.png')
plt.close()

# writing cosine distance values
f = open(path+'cosine_distances_dir7.dat', 'w')
for i in range(len(cosine_distance)):
	f.write(str(consine_distance[i])+'\n')
f.close()

print("Plots completed and saved")

# -----------------------------------------------------------------------
