'''
Face Emotion Detection using pictures
''''

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img = cv2.imread('4.jpg')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:     
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
predictions = DeepFace.analyze(img)
print(predictions)
font = cv2.FONT_HERSHEY_SIMPLEX

print('Dominant emotion is ')
cv2.putText(img,
                predictions[0]['dominant_emotion'],
                (190, 25),
                font, 0.75,
                (255, 0, 0),
                2,
                cv2.LINE_4);

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


