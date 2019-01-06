### Read Training data

import csv
import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.models import Model
import matplotlib.pyplot as plt

lines = []
with open('../data2019/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# ignore the csv header
lines.pop(0)

# Put the filename and measurement into seperated array.
images = []
measurements = []
for line in lines:
    source_path = line[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = '../data2019/IMG/' + filename
    image = cv2.imread(local_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3]) * 1.5  # Amplify the measurement.
    measurements.append(measurement)

print("images: " + str(len(images)))
print("measurements: " + str(len(measurements)))

## Flip the images
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

print("augmented_images: " + str(len(augmented_images)))
print("augmented_measurements: " + str(len(augmented_measurements)))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


### Define model and train.
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))) ## Normalize input.
model.add(Cropping2D(cropping=((70,25),(0,0))))

## Nvidia end to end pipeline
# convolution neural network 
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))

# convolution neural network 
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))


# convolution neural network 
model.add(Convolution2D(128,3,3,activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, verbose=1)

model.save('model2019.h5')


### Visual train model.
# history_object = model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=1)
# history_object = model201901.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data = 
#     validation_generator,
#     nb_val_samples = len(validation_samples), 
#     nb_epoch=5, verbose=1)

### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
# plt.savefig("training.png")