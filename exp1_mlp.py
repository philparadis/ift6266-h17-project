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

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras import losses

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
    flat = np.copy(img_array)
    flat = flat.flatten()
    img_top = flat[0:64*16,:]
    img_bottom = flat[64*48:64*64,:]
    no_middle_cols = np.delete(img_array, range(16,48))
    img_middle = no_middle_cols.flatten()[32*16:32*48]

    final_img = np.concatenate((img_top, img_middle, img_bottom), axis=0)
    print("outer_frame shape = ", np.shape(final_img))
    return final_img
    
for i, img_path in enumerate(train_images_paths):
    img = Image.open(img_path)
    img_array = np.array(img)

    # File names look like this: COCO_train2014_000000520978.jpg
    cap_id = os.path.basename(img_path)[:-4]

    ### Get input/target from the images
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
    if len(img_array.shape) == 3:
        X_outer = np.copy(img_array)
        X_outer_mask = np.array(np.ones(np.shape(img_array)), dtype='bool')
        X_outer_mask[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = False
        zipped = zip(X_outer, X_outer_mask)
        X_outer = X_outer.flatten()
        X_outer_mask = X_outer_mask.flatten()
        X_outer = X_outer[X_outer_mask]
        
        X_inner = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        X_inner = X_inner.flatten()
    else:
        continue
        #X_outer = np.copy(img_array)
        #X_outer[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
        #X_inner = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

    
    #Image.fromarray(img_array).show()
    X_train_inner.append(X_inner)
    X_train_outer.append(X_outer)
    captions = np.array([cap_id] + caption_dict[cap_id])
    X_train_caption.append(captions)

X_train_inner = np.array(X_train_inner)
X_train_outer = np.array(X_train_outer)
X_train_caption = np.array(X_train_caption)

print("Finished loading full dataset...")
print("X_train_inner shape   = ", np.shape(X_train_inner))
print("X_train_outer shape   = ", np.shape(X_train_outer))
print("X_train_caption shape = ", np.shape(X_train_caption))

print("First 3 rows and first 10 pixels of X_train_inner:")
print(X_train_inner[0, range(10)])
print(X_train_inner[1, range(10)])
print(X_train_inner[2, range(10)])

input_dim = 64*64*3 - 32*32*3
output_dim = 32*32*3
batch_size = 128
num_epochs = 10

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
X_train, X_test, Y_train, Y_test = train_test_split(X_train_outer,
                                                    X_train_inner,
                                                    test_size=0.20,
                                                    random_state=1)

print("Splitting dataset into training and testing sets with shuffling...")
print("X_train.shape = ", X_train.shape)
print("X_test.shape  = ", X_test.shape)
print("Y_train.shape = ", Y_train.shape)
print("Y_test.shape  = ", Y_test.shape)

#some preprocessing
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# Y_train = Y_train.astype('float32')
# Y_test = Y_test.astype('float32')
# X_train /= 255
# X_test /= 255
# Y_train /= 255
# Y_test /= 255

print("Creating MLP model...")
# Create model
model = Sequential()
model.add(Dense(units=2000, input_shape=(input_dim, )))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=(1000)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=output_dim))
model.add(Activation('relu'))

# Print model summary
print("Model summary:")
print(model.summary())

print("Compiling model...")
# Compile model
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss=losses.mean_absolute_error, optimizer='adam', metrics=['accuracy'])

print("Fitting model...")
# Fit the model
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size)

# evaluate the model
scores = model.evaluate(X_train, Y_train, batch_size=batch_size)
print("Training score %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
print("Testing score %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#%% Save model
print("Saving model as 'last_model.h5'")
model.save('last_model.h5')

#%% Load model
print("Loading model from disk 'last_model.h5'...")
model = load_model('last_model.h5')

X_test_predict = model.predict(X_test, batch_size=batch_size)

num_rows = X_test_predict.shape[0]
X_test_predict = np.reshape(X_test_predict, (num_rows, 32, 32, 3))

num_rows = Y_test.shape[0]
Y_test = np.reshape(Y_test, (num_rows, 32, 32, 3))

for row in range(10):
    img = Image.fromarray(X_test_predict[row,:,:,:])
    img.show()
    img.save('X_test_predict_' + str(row) + '.jpg')

    img = Image.fromarray(Y_test[row,:,:,:])
    img.show()
    img.save('Y_test_' + str(row) + '.jpg')
