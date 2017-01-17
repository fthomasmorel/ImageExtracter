from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse
import cv2

def image_to_feature_vector(image, size=(32, 32)):
    return cv2.resize(image, size)

def cropImage(image, margin = 0.05, resizing=300):
    _, w, h = image.shape[::-1]

    ratio = resizing/w
    if ratio < 1:
        image = cv2.resize(image, (int(round(ratio * w)), int(round(ratio * h))))

    _, w, h = image.shape[::-1]

    margin = w*0.05
    top = 0
    bottom = h
    left = 0
    right = w

    for i in range(0, h-1):
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

    for i in range(0, w-1):
        line_sum = sum([val for sublist in image[:,i].tolist() for val in sublist])
        if line_sum < h*3*255:
            left = max(i-margin, 0)
            break

    for i in range(0, w):
        i = w-i-1
        line_sum = sum([val for sublist in image[:,i].tolist() for val in sublist])
        if line_sum < h*3*255:
            right = min(i+margin,w)
            break

    if bottom-top > right-left:
        left = max(left + (right-left)/2 - (bottom-top)/2, 0)
        right = min(left + (right-left)/2 + (bottom-top)/2, w)
        if right == w:
            left = right-(bottom-top)
        if left == 0:
            right = left+(bottom-top)
        return image[int(top):int(bottom),int(left):int(right)]
    else:
        top = max(top + (bottom-top)/2 - (right-left)/2, 0)
        bottom = min(top + (bottom-top)/2 + (right-left)/2, h)
        if bottom == h:
            top = bottom-(right-left)
        if top == 0:
            bottom = top+(right-left)
        return image[int(top):int(bottom),int(left):int(right)]

def createModel(model_file):
    classes = ['bomb', 'danger', 'fire', 'car', 'forbidden', 'flash', 'dead', 'air', 'water', 'urgent', 'flag', 'warning', 'parking', 'man']

    # Make sure the data is normalized
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Define our network architecture:

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, 32, 32, 3],
    data_preprocessing=img_prep,
    data_augmentation=img_aug)

    # Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='relu')

    # Step 2: Max pooling
    network = max_pool_2d(network, 2)

    # Step 3: Convolution again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 4: Convolution yet again
    network = conv_2d(network, 64, 3, activation='relu')

    # Step 5: Max pooling again
    network = max_pool_2d(network, 2)

    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, 0.5)

    # Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
    network = fully_connected(network, len(classes), activation='softmax')

    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam',
    loss='categorical_crossentropy',
    learning_rate=0.001)

    # Wrap the network in a model object
    model = tflearn.DNN(network)
    model.load(model_file)
    return model


def processImage(imagePath):
    image = cv2.imread(imagePath,0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    _, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
    image = cropImage(image)
    return image_to_feature_vector(image)


def classify(model, image):
    classes = ['bomb', 'danger', 'fire', 'car', 'forbidden', 'flash', 'dead', 'air', 'water', 'urgent', 'flag', 'warning', 'parking', 'man']

    dictionary = {}
    result = {}

    data = np.array([image]) / 255.0
    prediction = model.predict(data)[0]

    for i in range(0, len(prediction)):
        dictionary[sorted(classes)[i]] = prediction[i]

    dictionary = sorted(dictionary.items(), key=lambda x: -x[1])
    for _, (x,y) in enumerate(dictionary):
        result[x] = y

    return result

#
# parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
# parser.add_argument('image', type=str, help='The image image file to check')
# args = parser.parse_args()
#
# image = cv2.imread('/Users/fthomasmorel/Desktop/file.png')
# # cv2.imshow('test', image)
# # cv2.waitKey(0)
# cv2.imshow('test', cropImage(image))
# cv2.waitKey(0)
