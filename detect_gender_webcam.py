from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import time
# load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man','woman']
countFemale = 0
countMale = 0
# loop through frames
previousFaceCount  = 0
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    currFaceCount = len(face)
    if(currFaceCount!=previousFaceCount):
        cv2.putText(frame, str(countMale)+" Males and "+str(countFemale)+" Females", (80, 80),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        previousFaceCount = currFaceCount

    else:
        cv2.putText(frame, str(countMale)+" Males and "+str(countFemale)+" Females", (20, 20),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)


    # time
    start_time = time.time()

    elapsed_time = int(time.time() - start_time)
    print(elapsed_time)
    cv2.putText(frame, str(currFaceCount)+" overall count", (100, 50),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        cv2.putText(frame, str(face), (70,70),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 255, 0), 2)
        face_crop = np.expand_dims(face_crop, axis=0)


        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        flag = 0
        label = classes[idx]
        if(currFaceCount!=previousFaceCount):
            if(idx==1):
                countFemale += 1

            elif(flag==0):
                countMale += 1



        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        #cv2.putText(frame, str(countMale)+" female"+str(countFemale), (40, 40),  cv2.FONT_HERSHEY_SIMPLEX,
                    #0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
