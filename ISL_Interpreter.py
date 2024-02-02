# This is the part of the code responsible for Video Capturing and grayscaling and resizing.

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# Loading trained model
model = load_model("C:/Users/nandi/AppData/Local/Programs/Python/Python310/Hackathon/Sign language interpreter/ISL_Interpreter")

# Define the categories of hand signs
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z']

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 128, 128, 1))
    
    # Perform prediction
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)
    sign = CATEGORIES[predicted_class]
    
    # Display the predicted hand sign
    cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Hand Sign Recognition', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
