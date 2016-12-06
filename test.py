import cv2
import numpy as np
import requests
import operator
import json

img = cv2.imread('bank/00122.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
w, h = gray.shape[::-1]
ratio = w/2480

threshold = 0.88
template = cv2.imread('coins2.png',0)
w, h = template.shape[::-1]
ratio = 1
template = cv2.resize(template, (int(round(ratio * w)), int(round(ratio * h))))
res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
loc = np.where( res >= threshold)

pointDown = []
pointUp = []

for pt in zip(*loc[::-1]):
    x1 = pt[0]+w/2+10
    y1 = pt[1]+h/2+10
    res = filter(lambda (x, y): round(x/100) == round(x1/100) and round(y/100) == round(y1/100), pointUp)
    if len(res) == 0:
        pointUp.append((x1,y1))

if len(pointUp) > 30:
    for (x1,y1) in pointUp:
        picto = gray[y1:y1+ratio*230, x1:x1+ratio*230]
        _, picto = cv2.threshold(picto, 220, 255, cv2.THRESH_BINARY)
        cv2.imwrite('tmp.png', picto)
        r = requests.post('https://pictogram.thomasmorel.io/', files={'file': open('tmp.png', 'rb')})
        json_data = json.loads(r.text)
        classifier = sorted(json_data.items(), key=operator.itemgetter(1))
        classe, _ = classifier[-1]
        cv2.imshow(str(classe), picto)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
