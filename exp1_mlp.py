#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:54:01 2017

@author: paradiph
"""

import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
#from skimage.transform import resize

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation


#%% Experiment with MNIST dataset
#loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#some preprocessing
# y_train = np_utils.to_categorical(y_train,10)
# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# while 1:
#     for i in range(1875):
#         if i%125==0:
#             print "i = " + str(i)
#         yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]

#%%


# Assume this script is being run from the root directory of the git repo

#######################################
# Dataset
#######################################
# The data is already split into training and validation datasets
# The training dataset has:
# - 82782 items
# - 984 MB of data
# The validation dataset has:
# - 40504 items
# - 481 MB of data
#
# There is also a pickled dictionary that maps image filenames (minutes the
# .jpg extension) to a list of 5 strings (the 5 human-generated captions).
# This dictionary is an OrderedDict with 123286 entries.

root_dir = os.environ['HOME'] + '/git-repos/ift6266-h2017-project/'

path_mscoco_dataset="./datasets/mscoco_inpainting/inpainting/"
path_train="train2014"
path_val="val2014"
path_caption_dict="dict_key_imgID_value_caps_train_and_valid.pkl"

#%% PATHS
mscoco="datasets/mscoco_inpainting/inpainting/"
split="train2014"
caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"

#%% Load training images and captions
data_path = os.path.join(mscoco, split)
caption_path = os.path.join(mscoco, caption_path)
with open(caption_path) as fd:
    caption_dict = pkl.load(fd)

# Get a list of all training images full filename paths
print data_path + "/*.jpg"
train_images_paths = glob.glob(data_path + "/*.jpg")
#batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

#%% Create dataset containing the images pixel data

X_train_outer = []
X_train_inner = []
X_train_caption = []

def flatten_outer_frame(img_array, dim_outer=(64, 64), dim_inner=(32, 32)):
    flat = img_array.flatten()
    img_top = flat[0:64*16]
    img_bottom = flat[64*48:64*64]
    no_middle_cols = np.delete(img_array, range(16,48))
    img_middle = no_middle_cols.flatten()[64*16:64*32]

    return np.concatenate((img_top, img_middle, img_bottom), axis=0)
    
for i, img_path in enumerate(train_images_paths):
    img = Image.open(img_path)
    img_array = np.array(img)

    # File names look like this: COCO_train2014_000000520978.jpg
    cap_id = os.path.basename(img_path)[:-4]

    ### Get input/target from the images
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
    if len(img_array.shape) == 3:
        X_outer = np.copy(img_array)
        X_outer[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
        X_inner = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
    else:
        X_outer = np.copy(img_array)
        X_outer[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
        X_inner = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

    
    #Image.fromarray(img_array).show()
    X_train_inner += X_inner.flatten()
    X_train_outer += flatten_outer_frame(X_outer)
    captions = [cap_id] + caption_dict[cap_id]
    X_train_caption += np.array(cap_id, captions)

print(X_train_inner[range(10),range(5)])


X_train_inner = np.array(X_train_inner, dtype="float32")
X_train_outer = np.array(X_train_outer)
X_train_caption = np.array(X_train_caption)

print("Finished loading full dataset...")
print("X_train_inner shape   = ", X_train_inner.shape())
print("X_train_outer shape   = ", X_train_outer.shape())
print("X_train_caption shape = ", X_train_caption.shape())

input_dim = 32*32
output_dim = 64*64 - 32*32
batch_size = 128

# model = Sequential([
#     Dense(32, input_dim=input_dim),
#     Activation('relu'),
#     Dense(2048),
#     Activation('relu'),
#     Dense(output_dim)
#     Activation('softmax'),
# ])

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
#           verbose=1, validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# split into input (X) and output (Y) variables
from sklearn.cross_validation import train_test_split
data, labels = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, Y_train, Y_test = train_test_split(X_train_outer,
                                                    X_train_inner,
                                                    test_size=0.20,
                                                    random_state=1)

# create model
model = Sequential()
model.add(Dense(batch_size, input_dim=input_dim, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense((64*64*3-32*32*3)/2, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32*32*3, init='uniform', activation='relu'))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, nb_epoch=150, batch_size=batch_size)

# evaluate the model
scores = model.evaluate(X_train, Y_train)
print("Training score %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(X_test, Y_test)
print("Testing score %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

