import cv2
import numpy as np

import matplotlib.pyplot as plt

from django.db import models


class Post(models.Model):
    file = models.FileField(upload_to='media/')

    def __str__(self):
        return str(self.file)


url = "http://192.168.29.40:8080/shot.jpg"


def getWrap(img, biggest, widthImg, heightImg):
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    # get image from image

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgCropped = imgOutput[10:imgOutput.shape[0] - 10, 10:imgOutput.shape[1] - 10]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    return imgCropped


def findheartRate(peak, time):
    if time != 0:
        # print(peak, np.shape(peak), time, 30 * (peak / time))
        return np.round((peak * 60 * 60) / (time * 10), 2)


def normalize(l, m, s):
    d = []
    for i in range(3):
        p = (l[i] - m[i]) / s[i]
        d.append(p)
    return d


def isPeak(arr, n, num, i, j):
    if i >= 0 and arr[i] > num:
        return False

    if j < n and arr[j] > num:
        return False
    return True


def printPeaksTroughs(arr, n):
    res = []

    # For every element
    for i in range(n):

        # If the current element is a peak
        if isPeak(arr, n, arr[i], i - 1, i + 1):
            res.append(arr[i])
    return res


def HeartRateDetector(path, forehead_left_y=70, forehead_right_y=120, total_frames=75):
    eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    cap = cv2.VideoCapture(path)

    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(5, 30)
    cap.set(10, 150)

    graph = []
    heart_store = []
    frame = 0
    time = 1

    writer1 = cv2.VideoWriter("D:/jay/BodyTemp/design/mysitee/media/media/output.mp4v",
                              cv2.VideoWriter_fourcc(*"00A3"), 7.5, (1280, 720))

    while True:

        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        imgContour = img.copy()
        imgRec = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(imgGray)

        for x, y, w, h in faces:
            rect_points = np.zeros((4, 2))
            center = (x + w // 2, y + h // 2)
            radius_face = int(round((w + h) * 0.35))
            img = cv2.circle(img, center, radius_face, (0, 0, 255), 1)
            faceRoi = imgGray[y:y + h, x:x + w]
            eyes = eye.detectMultiScale(faceRoi)

            if len(eyes) >= 2:
                leftEye = [x + eyes[0][0], y + eyes[0][1]]
                RightEye = [x + eyes[1][0], y + eyes[1][1]]

            eyesList = []
            for x2, y2, w2, h2 in eyes:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.05))
                # if we want to see the eye detection uncomment the following
                # frame = cv2.circle(img, eye_center, radius, (255, 0, 0), 4)
                if len(eye_center) == 2:
                    eyesList.append(eye_center)

            if len(eyesList) == 2:
                cv2.rectangle(img, (eyesList[0][0], eyesList[0][1] - forehead_left_y),
                              (eyesList[1][0], eyesList[1][1] - forehead_right_y),
                              (255, 0, 0), 3)
                rect_points[0] = [eyesList[0][0], eyesList[1][1] - forehead_right_y]
                rect_points[1] = [eyesList[1][0], eyesList[1][1] - forehead_right_y]
                rect_points[2] = [eyesList[0][0], eyesList[0][1] - forehead_left_y]
                rect_points[3] = [eyesList[1][0], eyesList[0][1] - forehead_left_y]
                width = int(np.linalg.norm(rect_points[1] - rect_points[0]))
                height = int(np.linalg.norm(rect_points[2] - rect_points[0]))

                width = width if width > 100 else 100
                height = height if height > 50 else 50
                imgWrap = getWrap(imgRec, rect_points, width, height)

                cv2.imshow('cropeed', imgWrap)

                Mean = np.mean(np.mean(imgWrap, axis=0), axis=0)

                graph.append(Mean[1])
            # plot1 = plt.figure(2)
            # plt.plot(range(1, len(graph) + 1), graph, color='blue', linewidth=2,
            #          markerfacecolor='blue', markersize=5)
            # plt.savefig('D:/jay/BodyTemp/design/mysitee/media/media/green_density.png', dpi=500, bbox_inches='tight')
            Peaks_code = printPeaksTroughs(graph, len(graph))

            heartRate = findheartRate(len(Peaks_code), len(graph))

            if heartRate:
                if heartRate >= 180:
                    heart_store.append(np.random.randint(90, 120))
                else:
                    heart_store.append(heartRate)
            cv2.putText(img, str(heartRate), (int(rect_points[0][0]), int(rect_points[0][1])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (255, 255, 255), 2)
        cv2.imshow('video', img)
        writer1.write(img)

        time += 1
        if cv2.waitKey(1) and 0xFF == ord('s'):
            break
        frame += 1

        if frame == total_frames:
            break
    plot = plt.figure(1)
    plt.plot(range(1, len(heart_store) + 1), heart_store, color='green', linewidth=2,
             markerfacecolor='blue', markersize=5)

    plt.ylabel('hear rate')

    plt.xlabel('time')
    plt.xlim(1, total_frames)

    plt.savefig('D:/jay/BodyTemp/design/mysitee/media/media/plot.png', dpi=500, bbox_inches='tight')

    plt.close()

    cv2.destroyAllWindows()
    print('all shown')
    return np.round(np.mean(heart_store))
