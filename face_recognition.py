import cv2
import os
import csv
import sys
import numpy as np

# Face recognizer using LBPH face recognizer
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizerFont = cv2.FONT_HERSHEY_DUPLEX

facialId = 0

nameID = ['Unknown', 'Rish', 'John']

# Video Capture objecy with the device index of the camera
videoCapture = cv2.VideoCapture(0)

# Setting the hieght and the width of the videoCapture object
videoCapture.set(3, 480) # Width
videoCapture.set(4, 480) # Height

countRight, countWrong = 0, 0

minWidth = 0.1 * videoCapture.get(3)
minHeight = 0.1 * videoCapture.get(4)

while (True):
    ret, frame = videoCapture.read()

    if ret is False:
        break

    # convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(int(minWidth), int(minHeight))
    )

    # Check if there are any faces in the frame
    if len(faces) >= 0:

        # Gets the coordinate of the in the frame and predict the face in the frame
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), 2)
            facialId, confidence = faceRecognizer.predict(gray[y:y+h, x:x+w])
            
            # If confidence of the predicted face is > 45%, increase the count
            if (confidence > 45):
                facialId = nameID[facialId]
                confidence = '{0}%'.format(round(100-confidence))
                countRight += 1
                
            # False predictions
            else: 
                facialId = 'Unknown'
                confidence = '{0}%'.format(round(100-confidence))
                countWrong += 1

            # Put text over the rectangle with the facial id and the confidence of correctness
            cv2.putText(frame, str(facialId), (x-5, y-5), recognizerFont, 1, (255,0,0), 2)
            cv2.putText(frame, str(confidence), (x, y+30), recognizerFont, 1, (255,0,0), 1)

        cv2.imshow('Recognize', frame)

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

videoCapture.release()
cv2.destroyAllWindows()

            
            
