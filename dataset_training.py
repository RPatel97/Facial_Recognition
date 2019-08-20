import os
import sys
import cv2
import numpy as np
from PIL import Image

# Path to dataset
filePath = 'dataset'

# Path to training output
myPath = 'trainer'

# To check if the training directory for the output exists if not, create it
if not os.path.isdir(myPath):
    os.makedirs(myPath)

# Local Binary Pattern Histogram for face
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Gets the image with the labels
def getLabeledImage(filePath):

    # Gets all the images from the dataset folder 
    imagePaths = [os.path.join(filePath,x) for x in os.listdir(filePath)]

    faceSamples = []
    facialId = []

    # Gets the lable of the image and puts it in an array
    for imgPath in imagePaths:
        img = Image.open(imgPath)
        img_arr = np.array(img, 'uint8')
        ids = int(os.path.split(imgPath)[-1].split('.')[1])

        faces = faceCascade.detectMultiScale(img_arr)

        # Gets the coordinates of the faces and the id assigned to the face
        for (x,y,w,h) in faces:
            faceSamples.append(img_arr[y:y+h,x:x+w])
            facialId.append(ids)

    return faceSamples, facialId

print('\nTraining faces . . .')

# Trains the face recognizer 
faces, ids = getLabeledImage(filePath)
faceRecognizer.train(faces, np.array(ids))

# Writes all the training data to the file
faceRecognizer.write('trainer/trainer.yml')

print('\n{0} face(s) trained.'.format(len(np.unique(ids))))