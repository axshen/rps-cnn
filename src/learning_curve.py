#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


history_file = os.getenv('TRAIN_HISTORY', os.path.join(os.path.dirname(__file__), 'records/history'))

with open(history_file, 'rb') as f:
    history = pickle.load(f)

loss = np.array(history['loss'])
val_loss = np.array(history['val_loss'])
epoch = np.arange(len(loss))

c1 = (42/255., 122/255., 181/255.)
c2 = (43/255., 149/255., 83/255.)

plt.plot(epoch, loss, color=c1, label='training loss')
plt.plot(epoch, val_loss, color=c2, label='validation loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
# plt.show()
plt.savefig("learning_curve.pdf")
