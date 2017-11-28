import csv
import cv2
import numpy as np
import math

import math
    
from find_lane import *

lines  = []
with open('sample_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'sample_data/IMG/' + filename
        image = cv2.imread(current_path)
        image_w_lines = find_lane(image)
        measurement = float(line[3])
        if i == 1:
            measurement += 0.25
        elif i == 2:
            measurement -= 0.25
        if np.absolute(measurement) < 0.1:
            if np.random.uniform() < 0.7:
                continue
        if np.absolute(measurement) < 0.2:
            if np.random.uniform() < 0.2:
                continue
        images.append(image_w_lines)
        measurements.append(measurement)

augemented_images, augemented_measurements = [], []
for image, measurement in zip(images, measurements):
    augemented_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
    augemented_measurements.append(measurement)

    # Flip
    image_flipped = cv2.flip(image, 1)
    augemented_images.append(cv2.cvtColor(image_flipped, cv2.COLOR_BGR2YUV))
    augemented_measurements.append(measurement * -1.0)

    # Random brightness
    image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image_hsv = np.array(image_hsv, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image_hsv[:,:,2] = image_hsv[:,:,2]*random_bright
    image_hsv[:,:,2][image_hsv[:,:,2]>255]  = 255
    image_brightness = np.array(image_hsv, dtype = np.uint8)
    image_brightness = cv2.cvtColor(image_brightness,cv2.COLOR_HSV2BGR)
    image_brightness = cv2.cvtColor(image_brightness,cv2.COLOR_BGR2YUV)
    augemented_images.append(image_brightness)
    augemented_measurements.append(measurement)

    # Random shadow
    image_shadow = np.array(image, dtype = np.float64)
    h,w = image_shadow.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        image_shadow[:,0:mid,:] *= factor
    else:
        image_shadow[:,mid:w,:] *= factor
    image_shadow = np.array(image_shadow, dtype = np.uint8)
    augemented_images.append(cv2.cvtColor(image_shadow, cv2.COLOR_BGR2YUV))
    augemented_measurements.append(measurement)

    # Random shift
    image_shift = image
    h,w,_ = image_shift.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    image_shift = cv2.warpPerspective(image_shift,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    augemented_images.append(cv2.cvtColor(image_shift, cv2.COLOR_BGR2YUV))
    augemented_measurements.append(measurement)

X_train = np.array(augemented_images)
y_train = np.array(augemented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam
from keras import regularizers

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(24, kernel_size=(5,5), strides=(2, 2), activation='elu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(36, kernel_size=(5,5), strides=(2, 2), activation='elu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(48, kernel_size=(5,5), strides=(2, 2), activation='elu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(64, kernel_size=(3,3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(64, kernel_size=(3,3), activation='elu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Flatten())
model.add(Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.5))
model.add(Dense(50, activation='elu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='elu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))

adam = Adam(lr = 0.0001)
model.compile(loss='mse', optimizer=adam)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15)

model.save('model.h5')