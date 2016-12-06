import numpy as np
import argparse
import cv2

def cropImage(image, margin = 0.05):
    _, w, h = image.shape[::-1]

    margin = w*0.05
    top = 0
    bottom = h
    left = 0
    right = w

    for i in range(0, h):
        line_sum = sum([val for sublist in image[i].tolist() for val in sublist])
        if line_sum < w*3*255:
            top = max(i-margin,0)
            break

    for i in range(0, h):
        i = h-i-1
        line_sum = sum([val for sublist in image[i].tolist() for val in sublist])
        if line_sum < w*3*255:
            bottom = min(i+margin, h)
            break

    for i in range(0, w):
        line_sum = sum([val for sublist in image[:,i].tolist() for val in sublist])
        if line_sum < h*3*255:
            left = max(i-margin, 0)
            break

    for i in range(0, w):
        i = h-i-1
        line_sum = sum([val for sublist in image[:,i].tolist() for val in sublist])
        if line_sum < h*3*255:
            right = min(i+margin,w)
            break

    if bottom-top > right-left:
        left = w/2 - (bottom-top)/2
        right = w/2 + (bottom-top)/2
        img = image[top:bottom,left:right]
    else:
        top = h/2 - (right-left)/2
        bottom = h/2 + (right-left)/2
        img = image[top:bottom,left:right]

    cv2.imshow('test', img)
    cv2.waitKey(0)
