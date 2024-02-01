import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Load the data
X = pickle.load(open("C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Notebooks/X.pickle", "rb"))
y = pickle.load(open("C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Notebooks/y.pickle", "rb"))

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
model.add(Dense(1, activation='sigmoid'))  # Added activation directly in Dense layer

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
