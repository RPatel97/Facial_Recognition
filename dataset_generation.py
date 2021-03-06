# Important imports
import numpy as np
import cv2
import os
import sys

# Video Capture objecy with the device index of the camera
videoCapture = cv2.VideoCapture(0)

# Setting the hieght and the width of the videoCapture object
videoCapture.set(3, 480) # Width
videoCapture.set(4, 480) # Height

# Haar cascade for face and eyes
facialCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# to get a face id for every user
facialId = input('\n Enter faceID or the name of a user:\n')

count = 0

# create a directory for dataset
myPath = 'dataset'

# check if dataset directory exists already, if not create one
if not os.path.isdir(myPath):
    os.makedirs(myPath)

while(True):

    # Capture video frame-by-frame
    ret, frame = videoCapture.read()

    # To grayscale the frames that are returned
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facialCascade.detectMultiScale(
        frame, 
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (20,20)
    )

    # To find the face(s) in the frame captured by the videocapture object
    if len(faces) >= 0:
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            
            count += 1

            # Region of image in grayscale and color
            roi_gray = grayscale[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # To find eyes in the face
            eyes = eyeCascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

            cv2.imwrite('dataset/user.' + str(facialId) + '.' + str(count) + '.jpg', grayscale[y:y+h,x:x+w])

        cv2.imshow('Face detection', frame)

        k = cv2.waitKey(30) & 0xff
        # 'ESC' key to quit
        if k == 27:
            break
        elif count >= 1000:
            break

videoCapture.release()
cv2.destroyAllWindows()
