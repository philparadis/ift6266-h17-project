#!/usr/bin/env python2
# coding: utf-8

import os
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image

import settings

### Define utility functions
def normalize_data(data):
    if data.dtype == 'float32':
        return
    data = data.astype('float32')
    data /= 255
    return data

def denormalize_data(data):
    if data.dtype == 'uint8':
        return
    data *= 255
    data = data.astype('uint8')
    return data

def transpose_colors_channel(data):
    if len(data.shape) != 4:
        raise ValueError("Dataset is not a 4-tensor as expected.")
    if data.shape[3] != 3:
        raise ValueError("Colors channel is not 3-dimensional as expected.")
    width = data.shape[1]
    height = data.shape[2]
    return data.transpose(0, 3, 1, 2).reshape(data.shape[0], 3, width, width)

### Define the main class for handling our dataset called InpaintingDataset

class InpaintingDataset(object):
    def __init__(self, input_dim, output_dim, colors_channel_first = True):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.images = []
        self.captions_ids = []
        self.captions_dict = []

        self.images_inner_flat = []
        self.images_outer_flat = []

        self.images_outer2d = []
        self.images_inner2d = []

        self._colors_channel_first = colors_channel_first
        self._is_dataset_loaded = False
        self._num_rows = None

        self.X = None
        self.Y = None
        self.id_train = []
        self.id_test = []

    def load_dataset(self, force_reload = False):
        self._read_jpgs_and_captions(force_reload = force_reload)
        self._num_rows = self.images.shape[0]
    
    def _read_jpgs_and_captions(self, force_reload = False):
        # Check if 'npy' dataset already exists
        if force_reload == True:
            self._is_dataset_loaded = False
        # Check if the dataset has been loaded already and saved to the '.npy' format. 
        elif (    os.path.isfile(os.path.join(settings.MSCOCO_DIR, "images.npy")) == True
                  and os.path.isfile(os.path.join(settings.MSCOCO_DIR, "captions_ids.npy"))
                  and os.path.isfile(os.path.join(settings.MSCOCO_DIR, "captions_dict.npy"))
             ):
            self._load_jpgs_and_captions_npy()

        if self._is_dataset_loaded == False:
            # Get a list of all training images full filename paths
            printf("(!) Loading dataset from individual JPG files and pickled dictionaries...")
            print(" * Training images paths     = " + settings.TRAIN_DIR + "*.jpg")
            train_images_paths = glob.glob(settings.TRAIN_DIR + "/*.jpg")
            num_train_images_path = len(train_images_paths)
            print(" * Number of training images =  %i" % num_train_images_path)
            print("Loading images and captions into memory...")
            print("")
            
            with open(settings.CAPTIONS_PKL_PATH) as fd:
                caption_dict = pkl.load(fd)

            images = []
            captions_ids = []
            captions_dict = []
            nb_images_paths = len(train_images_paths)
            for i, img_path in enumerate(train_images_paths):
                img = Image.open(img_path)
                img_array = np.array(img)

                # File names look like this: COCO_train2014_000000520978.jpg
                cap_id = os.path.basename(img_path)[:-4]

                ### Get input/target from the images
                center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
                if len(img_array.shape) != 3:
                    # For now, discard greyscale images
                    continue

                images.append(img_array)
                captions_ids.append(cap_id)
                captions_dict.append(caption_dict[cap_id])
                
                if i % 5000 == 0:
                    print("Loaded image #%i" % i)
            print("Loaded last image #%i" % i)
            print(" * Number of color images loaded        = {}".format(images.shape[0]))
            print(" * Number of greyscale images discarded = {}".format(num_train_images_path - images.shape[0]))
            self.images = np.array(images)
            # Transpose image's color channel to be first
            self.images = transpose_colors_channel(self.images)
            #self.images = self.images.transpose(0, 3, 1, 2).reshape(self.images.shape[0], 3, 64, 64)
            self.captions_ids = np.array(captions_ids)
            self.captions_dict = np.array(captions_dict)
            
            self._is_dataset_loaded = True
            # Save dataset as npy file so that loading can be sped up in the future
            self._save_jpgs_and_captions_npy()

    def _load_jpgs_and_captions_npy(self):
        print("(!) Found project dataset encoded as'.npy' files on disk...")
        for i, filename in enumerate(["images.npy", "captions_ids.npy", "captions_dict.npy"]):
            path = os.path.join(settings.MSCOCO_DIR, filename)
            if i == 0:
                print("Loading training images: {}".format(path))
                self.images = np.load(path)
            elif i == 1:
                print("Loading captions ids: {}".format(path))
                self.captions_ids = np.load(path)
            elif i == 2:
                print("Loading captions dictionary: {}".format(path))
                self.captions_dict = np.load(path)
        self._is_dataset_loaded = True
        
    def _save_jpgs_and_captions_npy(self):
        for i, filename in enumerate(["images.npy", "captions_ids.npy", "captions_dict.npy"]):
            path = os.path.join(settings.MSCOCO_DIR, filename)
            print(" '{}'".format(path))
            if i == 0:
                print("Writing to disk training images: {}".format(path))
                np.save(path, self.images)
                self.images = np.load(path)
            elif i == 1:
                print("Writing to disk captions ids: {}".format(path))
                np.save(path, self.captions_ids)
                self.captions_ids = np.load(path)
            elif i == 2:
                print("Writing to disk captions dictionary: {}".format(path))
                np.save(path, self.captions_dict)
                self.captions_dict = np.load(path)

    def preprocess(self):
        print("Preprocessing {0} images for the '{1}' model...".format(len(self.images), settings.MODEL))

        images_outer_flat = []
        images_inner_flat = []
        images_outer2d = []
        images_inner2d = []

        # Don't forget we transposed the image, so the format of
        # 'img_array' will be (3, 64, 64).
        for i, img_array in enumerate(self.images):
            ### Get input/target from the images
            center = (int(np.floor(img_array.shape[1] / 2.)), int(np.floor(img_array.shape[2] / 2.)))
            if len(img_array.shape) != 3 and img_array.shape[0] != 3:
                raise ValueError("The image #{} does not have 3 color channels.".format(i))
        
            image = np.copy(img_array)

            if settings.MODEL == "mlp" or settings.MODEL == "test":
                outer = np.copy(img_array)
                outer_mask = np.array(np.ones(np.shape(img_array)), dtype='bool')
                outer_mask[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = False
                outer_flat = outer.flatten()
                outer_mask_flat = outer_mask.flatten()
                outer_flat = outer_flat[outer_mask_flat]

                inner = np.copy(img_array)
                inner = inner[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
                inner_flat = inner.flatten()

                images_outer_flat.append(outer_flat)
                images_inner_flat.append(inner_flat)
            elif settings.MODEL == "conv_mlp":
                outer_2d = np.copy(img_array)
                outer_2d[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
                inner2d = np.copy(img_array)
                inner2d = inner2d[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
                images_outer2d.append(outer_2d)
                images_inner2d.append(inner2d)
            elif settings.MODEL == "dcgan" or settings.MODEL == "wgan" or settings.MODEL == "lsgan":
                inner2d = np.copy(img_array)
                inner2d = inner2d[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
                images_inner2d.append(inner2d)

        if settings.MODEL == "mlp" or settings.MODEL == "test":
            self.images_inner_flat = np.array(images_inner_flat)
            self.images_outer_flat = np.array(images_outer_flat)
            self.images_outer_flat = normalize_data(self.images_outer_flat)
            self.images_inner_flat = normalize_data(self.images_inner_flat)
        elif settings.MODEL == "conv_mlp":
            self.images_outer2d = np.array(images_outer2d)
            self.images_inner2d = np.array(images_inner2d)
            self.images_inner2d = normalize_data(self.images_inner2d)
            self.images_outer2d = normalize_data(self.images_outer2d)
        elif settings.MODEL == "dcgan" or settings.MODEL == "wgan" or settings.MODEL == "lsgan":
            self.images_inner2d = np.array(images_inner2d)
            self.images = normalize_data(self.images)
            self.images_inner2d = normalize_data(self.images_inner2d)

    def postprocess(self):
        if settings.MODEL == "mlp" or settings.MODEL == "test":
            self.images_outer_flat = denormalize_data(self.images_outer_flat)
            self.images_inner_flat = denormalize_data(self.images_inner_flat)
        elif settings.MODEL == "conv_mlp":
            self.images_inner2d = denormalize_data(self.images_inner2d)
            self.images_outer2d = denormalize_data(self.images_outer2d)
        elif settings.MODEL == "dcgan" or settings.MODEL == "wgan" or settings.MODEL == "lsgan":
            self.images = denormalize_data(self.images)
            self.images_inner2d = denormalize_data(self.images_inner2d)

    def preload(self, test_size = 0.2, seed = 0):
        if settings.MODEL == "mlp" or settings.MODEL == "test":
            x = self.images_outer_flat
            y = self.images_inner_flat
            rand_seed = 1000 + seed
        elif settings.MODEL == "convmlp":
            x = self.images_outer2d
            y = self.images_inner2d
            rand_seed = 1001 + seed
        elif settings.MODEL == "dcgan" or settings.MODEL == "wgan" or settings.MODEL == "lsgan":
            x = self.images
            y = self.images_inner2d
            rand_seed = 1002 + seed
        else:
            raise Exception("You need to specify a model for the InpaintingDataset object using 'use_model(...)'.")
            

        ### Split into training and testing data
        from sklearn.model_selection import train_test_split  
        indices = np.arange(self.images.shape[0])
        id_train, id_test = train_test_split(indices,
                                             test_size=test_size,
                                             random_state=rand_seed)
        print("len(id_train) = {0}\nlen(id_test) = {1}".format(len(id_train), len(id_test)))
        
        ### Generating the training and testing datasets (80%/20% train/test split)
        print("Splitting dataset into training and testing sets with shuffling...")
        #X_train, X_test, Y_train, Y_test = x[id_train], x[id_test], y[id_train], y[id_test]

        self.X = x
        self.Y = y
        self.id_train = id_train
        self.id_test = id_test

        print("Preloading complete.")
        print("Input training dataset X has shape:  {0}".format(str(self.X[id_train].shape)))
        print("Output training dataset Y has shape: {0}".format(str(self.Y[id_train].shape)))
        print("Input training dataset X has shape:  {0}".format(str(self.X[id_test].shape)))
        print("Output training dataset Y has shape: {0}".format(str(self.Y[id_test].shape)))


    def return_data(self):
        return self.X[self.id_train,], self.X[self.id_test,], \
            self.Y[self.id_train,], self.Y[self.id_test,], \
            self.id_train, self.id_test

    def get_data(self, X = False, Y = False, Train = False, Test = False):
        if X and Y:
            raise Exception("Must specify either X=True or Y=True, but not both.")
        if Train and Test:
            raise Exception("Must specify either Train=True or Test=True, but not both.")
        if X and Train:
            return self.X[self.id_train,]
        if X and Test:
            return self.X[self.id_test,]
        if Y and Train:
            return self.Y[self.id_train,]
        if Y and Test:
            return self.Y[self.id_test,]
        raise Exception("Must specify one of X or Y as True and one of Train or Test as True.")
        
