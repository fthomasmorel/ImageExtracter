from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import argparse
from imutils import paths
import cv2
import os
#import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import croper

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	image = croper.crop(image)
	return cv2.resize(image, size)#.flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	if ".png" not in imagePath:
		continue
		
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split("_")[0]

	# construct a feature vector raw pixel intensities, then update
	# the data matrix and labels list
	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(label)

	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

classes = list(set(labels))
print(classes)

#encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)


# scale the input image pixels to the range [0, 1], then transform
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
# is set to `1` and all other entries to `0`

data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, len(classes))

print("[INFO] constructing training/testing split...")
(X, X_test, Y, Y_test) = train_test_split(
	data, labels, test_size=0.25, random_state=42)

# # # Load the data set
# X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))
# #
# # # Shuffle the data
# X, Y = shuffle(X, Y)

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
model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path='picto-classifier.tfl.ckpt')

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
		  show_metric=True, batch_size=96,
		  snapshot_epoch=True,
		  run_id='picto-classifier')

# Save model when training is complete to a file
model.save("picto-classifier.tfl")
print("Network trained and saved as picto-classifier.tfl!")
