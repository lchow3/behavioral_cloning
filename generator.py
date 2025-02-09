import os
import csv
import numpy as np
import random
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./data-3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './sim_data/data-3/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train,y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format
batch_size = 32
# Preprocess incoming data, centered around zero with small standard deviation
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 -.5, input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=(len(train_samples)//batch_size)*batch_size, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('generator.h5')

from keras.utils.visualize_util import plot
plot(model, to_file='generator.png')
