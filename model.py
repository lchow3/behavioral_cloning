import os
import csv
import cv2
import numpy as np
import random
import sklearn
import matplotlib.pyplot as plt

lines = []

with open('./sim_data/dirt-2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './sim_data/dirt-2/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    thresh = np.random.random()
    if (measurement == 0.0 and thresh < 0.5):
        continue
    else:
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print(X_train.shape)
print(y_train.shape)

import cv2
import numpy as np
import sklearn


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

# First Model
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 -.5, input_shape=(160,320,3)))
# model.add(Convolution2D(6,5,5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# Nvidia Model
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
model.fit(X_train, y_train, validation_split=0.25, shuffle=True, nb_epoch=3)

from keras.models import Model
import matplotlib.pyplot as plt

model.save('model.h5')

# save visualization of the network
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

model.summary()
