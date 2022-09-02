import csv
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import imutils
import datetime
import smtplib, ssl
port = 587  # For starttls
smtp_server = "smtp.gmail.com"
sender_email = "facemaskalert11@gmail.com"
# nm=""
# em=str(nm+"@gmail.com").lower()
receiver_email = ""
password = "Darshan@11"
import keyboard
from datetime import date


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f",
                "--face",
                type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m",
                "--model",
                type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join(
    [args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")


# from PIL import ImageGrab
path = 'face_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def sendEmail(name):
    str1=''.join([i for i in name if not i.isdigit()])
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        receiver_email=str(name+"@gmail.com").lower()
        server.sendmail(sender_email, receiver_email, str1+ ", You are not wearing a mask. If you try to re-enter the campus without a mask, you have to pay 50 rupees fine ")


def markInExcel(name):
    str1=''.join([i for i in name if not i.isdigit()])
    with open('excel.csv', 'a',  newline='') as f:
        writer = csv.writer(f)
        today = date.today()
        todaytime=datetime.datetime.now()
        writer.writerow(['name','date','time'])
        
        writer.writerow([str1+ " has not weared mask",todaytime])





#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    name=""
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        if(withoutMask>mask):
            print("without mask")
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    newstring=''.join([i for i in name if not i.isdigit()])
                    cv2.putText(img, newstring, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    print(newstring)
                    nm=name
                    cv2.putText(img, newstring+"-"+label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                    markInExcel(newstring)
                    if keyboard.is_pressed('e'): 
                        print("E is pressed")
                        sendEmail(name)
        else:
            print(label)
            color = (0, 255, 0) if label == "Mask" else (0, 255, 0)
            cv2.putText(img, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        # include the probability in the label

           # markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)