#
#  extractor.py
#  ImageExtracter
#
#  Created by Florent THOMAS-MOREL on 10/26/16.
#  MIT Liscence

import os
import cv2
import math
import sys
import numpy as np
from matplotlib import pyplot as plt


########################################################
#                                                      #
#                      CONSTANT                        #
#                                                      #
########################################################

OUTPUT_DIR = './output2/'
BANK_IMG_FOLDER = './bank/'
TEMPLATES_IMG_FOLDER = './picto/'

WIDTH_BASE_SIZE = 2480
PICTOS_PER_LINES = 5
NB_PICTOS = 7 * PICTOS_PER_LINES


########################################################
#                                                      #
#                IMAGE PRE TREATMENT                   #
#                                                      #
########################################################

def emptyImage(img_rgb):
    _, empty_image = cv2.threshold(img_rgb, 120, 255, cv2.THRESH_BINARY)
    return empty_image


def detectShapes(empty_img, img, ratio, threshold):
    template = cv2.imread('square.png',0)
    w, h = template.shape[::-1]
    template = cv2.resize(template, (int(round(ratio * w)), int(round(ratio * h))))

    points = []
    shapes = []

    res = cv2.matchTemplate(empty_img,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        x1 = pt[0]
        y1 = pt[1]
        special_squares = filter(lambda (x, y): (float(x1) / float(x)) > 0.9 and (float(y1) / float(y)) > 0.9 and (float(y1) / float(y)) < 1.1 and (float(x1) / float(x)) < 1.1, points)
        if len(special_squares) == 0:
            picto = img[y1:y1+h, x1:x1+w]
            points.append((x1, y1))
            shapes.append((picto, (x1, y1)))

    return shapes


########################################################
#                                                      #
#                   PICTO TREATMENT                    #
#                                                      #
########################################################

def getTemplates(path):
    templates = {}
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            pictoName = filename[:-4]
            templates[pictoName] = cv2.imread(imgPath, 0)

    return templates


def recognizePictogram(pictoZone, ratio):
    templates = getTemplates(TEMPLATES_IMG_FOLDER)
    for threashold in range(95, 50, -5):
        threashold = float(threashold)/100.0
        for template, tmpl_img in templates.items():
            w, h = tmpl_img.shape[::-1]
            tmpl_img = cv2.resize(tmpl_img, (int(round(ratio * w)), int(round(ratio * h))))
            res = cv2.matchTemplate(pictoZone, tmpl_img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threashold)
            for pt in zip(*loc[::-1]):
                return template

    return "-"

def pictogramZones(coords):
    x = min(set(map((lambda (x, _): x), coords)))
    ys = []
    for y in list(map((lambda (_, y): y), coords)):
        if len(filter(lambda y1: (float(y1) / float(y)) > 0.9 and (float(y) / float(y1)) < 1.1, ys)) == 0:
            ys.append(y)
    return [(x,y) for y in ys]


########################################################
#                                                      #
#                      DETECTION                       #
#                                                      #
########################################################

def classifyShapes(img, shapes, ratio):
    pictos = {}
    result = []
    heights = list(map((lambda (shape, _): shape.shape[::-1][1]), shapes))
    coords = list(map((lambda (_, coord): coord), shapes))
    height_avg = 0
    if len(heights) > 0:
        height_avg = int(np.mean(heights))
    else:
        return pictos
    zones = pictogramZones(coords)
    for (x,y) in zones:
        pictoZone = img[y:y+height_avg,1:x]
        result.append((recognizePictogram(pictoZone, ratio), y))
    for (name, y) in result:
        pictograms = list(map(lambda (shape, _): shape, list(filter(lambda (shape, (_, y1)): (float(y1) / float(y)) > 0.9 and (float(y1) / float(y)) < 1.1 , shapes))))
        if name not in pictos.keys():
            pictos[name] = pictograms
        else:
            pictos[name].extend(pictograms)
    return pictos


def processFile(file, threashold = 0.63):
    img = cv2.imread(file, 0)
    empty_img = emptyImage(img)
    width, height = empty_img.shape[::-1]
    ratio = float(min(width, WIDTH_BASE_SIZE)) / float(max(width, WIDTH_BASE_SIZE))
    shapes = detectShapes(empty_img, img, ratio, threashold)
    shapes = classifyShapes(img, shapes, ratio)
    errors = 0
    extracted = 0
    for key in shapes.keys():
        if key == "-":
            errors = len(shapes["-"])
            next
        extracted += len(shapes[key])
    missing = NB_PICTOS - (extracted + errors)
    return shapes, extracted, errors, missing


########################################################
#                                                      #
#                     EXPORTATION                      #
#                                                      #
########################################################

def writeImage(name, img, index):
    for dirname, _, _ in os.walk(OUTPUT_DIR):
        w, h = img.shape[::-1]
        img = img[20:h-20,20:w-20]
        #cv2.imwrite(dirname + name + "_" + str(index) + ".png", img)
        _, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        #cv2.imwrite(dirname + name + "_" + str(index) + "_dark.png", img)
        cv2.imwrite(dirname + name + "_" + str(index) + ".png", img)
        break


def exportImages(shapes, extracted, number):
    i = 0 + number
    for key in shapes.keys():
        if key != "-":
            for img in shapes[key]:
                #drawProgressBar(float(i)/float(extracted), 0)
                writeImage(key, img, i)
                i += 1


def drawProgressBar(percent, errorRate, barLen = 50):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[%s] %.2f%%, errors => %.2f%%" % (progress, percent * 100, errorRate))
    sys.stdout.flush()


########################################################
#                                                      #
#                        MAIN                          #
#                                                      #
########################################################

for dirname, _, filenames in os.walk(BANK_IMG_FOLDER):
    print "Analyzing " + str(len(filenames)) + " Images..."
    shapes = []
    done = 1
    errors = 0
    missings = 0
    progress = 0
    extracteds = 0
    messages = []
    for filename in filenames:
        drawProgressBar(float(progress)/float(len(filenames)), ((float(errors) + float(missings)) / float(done)) * 100)
        imgPath = os.path.join(dirname, filename)
        result, extracted, error, missing = processFile(imgPath)

        progress+= 1
        if extracted > NB_PICTOS:
            result, extracted, error, missing = processFile(imgPath, 0.65)

        if extracted > NB_PICTOS:
            messages.append("Unable to process [ " + filename +" ] : Found only " + str(extracted) + " images instead of " + str(NB_PICTOS))
            errors+= NB_PICTOS
        if extracted < NB_PICTOS and extracted > 15:
            messages.append("Unable to process [ " + filename +" ] : Found only " + str(extracted) + " images instead of " + str(NB_PICTOS))
            messages.append("Unable to process [ " + filename +" ] : Misdetected " + str(error) + " images out of " + str(extracted))
            errors += error
            missings += missing
        if extracted > 15:
            exportImages(result, extracted, extracteds)
            extracteds += extracted
            done += NB_PICTOS
            #shapes.append(result)
#        if progress > 5: break

    if done > 0 :
        drawProgressBar(1, ((float(errors) + float(missings)) / float(done)) * 100)
        print "\n"
        #print "Exporting " + str(extracteds) + " Images..."
        #drawProgressBar(1, 0)
        print "\n"
        print done, " pictograms treated"
        print errors, " pictograms misrecognized"
        print missings, " pictograms missing"
        print extracteds, " pictograms exported"
        print "Error rate: ", ((float(errors) + float(missings)) / float(done)) * 100, "%"
    else:
        print "[error] Unable to analyze the database"
