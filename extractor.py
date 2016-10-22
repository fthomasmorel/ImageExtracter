import math
import numpy as np
import cv2
from matplotlib import pyplot as plt


def rotateImage(image, threshold):
    angles = [x * 0.1 for x in range(-20, 20)]
    for angle in angles:
        img = image.copy()
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img = cv2.warpAffine(img,M,(cols,rows))

        tmpl_img = cv2.imread('cross.png',0)
        w, h = tmpl_img.shape[::-1]

        res = cv2.matchTemplate(img,tmpl_img,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        if len(zip(*loc[::-1])) == 2:
            return img
    return img




templates = ['flag', 'bomb', 'warning', 'flash', 'danger', 'car', 'fire']
points = []
shapes = []

img = cv2.imread('test7.png',0)
w, h = img.shape[::-1]
ratio = float(min(w,1448))/float(max(w,1448))
print ratio
img = cv2.resize(img, (int(round(ratio * w)), int(round(ratio * h))))
ret,gray = cv2.threshold(img,210,255,cv2.THRESH_TOZERO)
cv2.imshow('img',gray)

ret,thresh = cv2.threshold(gray,0,255,1)
_,contours,h = cv2.findContours(thresh,1,2)

i = 0
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx)==4:
        y1 = approx[0][0][1]+2
        x1 = approx[0][0][0]+2
        y2 = approx[2][0][1]-2
        x2 = approx[2][0][0]-2
        if math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2) > float(60)*ratio:
            special_squares = filter(lambda (x,y): (float(x1)/float(x)) > 0.9 and (float(y1)/float(y)) > 0.9 and (float(y1)/float(y)) < 1.1 and (float(x1)/float(x)) < 1.1, points)
            if len(special_squares) == 0:
                i+= 1
                points.append((x1,y1))
                cv2.imshow('img',gray)
                cv2.circle(gray,(x1,y1), 3, (0,0,255), 1)
                cv2.circle(gray,(x2,y2), 3, (0,0,255), 1)
                points.append((x1,y1))
                roi = img[y1:y2, x1:x2]
                shapes.append(roi)
                cv2.waitKey(0)
                cv2.imshow('roi',roi)
                if i % 5 == 0:
                    #cv2.circle(gray,(x1-int(float(220)*ratio),y1), 3, (0,0,255), 1)
                    #cv2.circle(gray,(x2-int(float(220)*ratio),y2), 3, (0,0,255), 1)
                    roi = img[y1:y2, 1:x1-4]
                    cv2.imshow('roi2',roi)
                    pattern = "-"
                    for template in templates:
                        tmpl_img = cv2.imread(template + '.png',0)
                        w, h = tmpl_img.shape[::-1]
                        tmpl_img = cv2.resize(tmpl_img, (int(round(ratio * w)), int(round(ratio * h))))
                        res = cv2.matchTemplate(roi,tmpl_img,cv2.TM_CCOEFF_NORMED)
                        threshold = 0.5
                        loc = np.where( res >= threshold)
                        for pt in zip(*loc[::-1]):
                            pattern = template
                            break

                        if pattern != "-":
                            break

                    index = 0
                    for shape in shapes:
                        filename = "/Users/fthomasmorel/Desktop/IRF/" + pattern + "_" + str(i) + "_" + str(index) + ".png"
                        cv2.imwrite(filename, shape)
                        print filename
                        index+=1

                    shapes = []
                    cv2.waitKey(0)

cv2.destroyAllWindows()
