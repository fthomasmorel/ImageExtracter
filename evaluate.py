import os
import cv2
import numpy as np
import requests
import operator
import json

def proceed(imagePath, threashold):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    w, h = gray.shape[::-1]
    ratio = w/2480

    template = cv2.imread('coins2.png',0)
    w, h = template.shape[::-1]
    ratio = 1
    template = cv2.resize(template, (int(round(ratio * w)), int(round(ratio * h))))
    res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threashold)

    pointDown = []
    pointUp = []

    for pt in zip(*loc[::-1]):
        x1 = pt[0]+w/2+10
        y1 = pt[1]+h/2+10
        res = filter(lambda (x, y): round(x/100) == round(x1/100) and round(y/100) == round(y1/100), pointUp)
        if len(res) == 0:
            pointUp.append((x1,y1))

    return len(pointUp)



threasholds = list(reversed([0.97, 0.95, 0.92, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.82, 0.80]))
found = 0
potential = 35
for dirname, _, filenames in os.walk('./bank/'):
    for filename in filenames:
        imgPath = os.path.join(dirname, filename)
        tries = 0

        num = proceed(imgPath, threasholds[0])
        while num > 35 and tries <= len(threasholds)-2:
            tries += 1
            num = proceed(imgPath, threasholds[tries])

        if num != 35:
            if "22.png" not in imgPath:
                print(imgPath + " | " + str(num) + " | " + str(tries) + " | " + str(threasholds[tries]))
                found += num
                potential += 35
        else:
            potential += 35

print(str(found))
print(str(potential))
