import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Read Data from MOV Files
# DATADIR = 'C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Data/MOVs/Adjectives'
DATADIR = 'C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Data/ISL_Dataset'
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # path to cats or dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        break
    break

IMG_SIZE = 128

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training_data = []

DATADIR = 'C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Data/ISL_Dataset'
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z']

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

import random
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Load the data
X = pickle.load(open("C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Sign language interpreter/X.pickle", "rb"))
y = pickle.load(open("C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Sign language interpreter/y.pickle", "rb"))

X = X / 255.0

# Build the model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1, activation='softmax'))  # Added activation directly in Dense layer

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, batch_size=1, epochs=10, validation_split=0.1)

