import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
from collections import deque

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
rate = engine.getProperty('rate')
engine.setProperty('rate', 125)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C"]

# Deque to store the last few predictions
prediction_history = deque(maxlen=3)
last_prediction = None

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            try:
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                if imgCrop.size == 0:
                    raise ValueError("Empty image crop detected")

                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                try:
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)

                    prediction_history.append(labels[index])

                    if len(prediction_history) == 3 and len(set(prediction_history)) == 1 and prediction_history[1]!=last_prediction:
                        last_prediction=labels[index]
                        engine.say(labels[index])
                        engine.runAndWait()
                        prediction_history = deque(maxlen=3)

                except Exception as e:
                    print("Error in classifier prediction:", e)

            except cv2.error as e:
                print("OpenCV error:", e)
            except ValueError as e:
                print(e)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)

    except Exception as e:
        print("Unexpected error:", e)
