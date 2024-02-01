# This is the part of the code responsible for Video Capturing and grayscaling and resizing.

import numpy as np
import cv2


IMG_SIZE = 128

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))

    cv2.imshow('frame', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
