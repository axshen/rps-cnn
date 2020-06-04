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

plt.plot(epoch, loss, color='blue', label='training loss')
plt.plot(epoch, val_loss, color='red', label='validation loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
