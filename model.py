# %%
import numpy as np
import cv2
import os

# Read Data from MOV Files
DATADIR = 'C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Data/MOVs'
CATEGORIES = os.listdir(DATADIR)

training_data = []

IMG_SIZE = 128

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Preprocess the frame (resize, normalize, convert to grayscale)
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            # Add the preprocessed frame and its label to training data
            training_data.append([resized_frame, category])
        cap.release()

# %%
import random
random.shuffle(training_data)

# %%
X = []
y = []

# %%
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# %%
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# %%
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
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)


