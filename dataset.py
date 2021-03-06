#!/usr/bin/env python2
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image

import settings
import utils
from utils import normalize_data, denormalize_data
from utils import force_symlink, get_json_pretty_print
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log
#from utils import save_keras_predictions, print_results_as_html
#from utils import unflatten_to_4tensor, unflatten_to_3tensor, transpose_colors_channel


### Define the main class for handling our dataset called InpaintingDataset

class BaseDataset(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self._images_filename = None
        self._test_images_filename = None
        self._captions_ids_filename = "captions_ids.npy"
        self._captions_dict_filename = "captions_dict.npy"

        self.images = []
        self.captions_ids = []
        self.captions_dict = []

        self.images_inner_flat = []
        self.images_outer_flat = []

        self.images_outer2d = []
        self.images_inner2d = []

        self.images_T = []
        self.images_inner_flat_T = []
        self.images_outer_flat_T = [] 

        self._is_dataset_loaded = False
        self._num_rows = None

        self.X = None
        self.Y = None
        self.id_train = []
        self.id_val = []

        self.test_images = []

    def transform_images(self, images):
        """Images should be a list of numpy arrays of the form (64, 64, 3). This function will turn it into a numpy batch 4-tensor of the form (batch_size, 64, 64, 3) or (batch_size, 3, 64, 64) depending on the implementation of the derived class."""
        pass

    def load_dataset(self, force_reload = False):
        self._read_jpgs_and_captions(force_reload = force_reload)
        self._num_rows = self.images.shape[0]

    def _try_read_npy_dataset(self):
        # Check if the dataset has been loaded already and saved to the '.npy' format.
        if not self._images_filename:
            raise Exception("ERROR: You did not define the filename the '.npy' dataset containing images.")

        prefix = ""
        if settings.MAX_TRAINING_SAMPLES != -1:
            prefix = "subset_"

        images_path = os.path.join(settings.MSCOCO_DIR, prefix + self._images_filename)
        test_images_path = os.path.join(settings.MSCOCO_DIR, prefix + self._test_images_filename)
        captions_ids_path = os.path.join(settings.MSCOCO_DIR, prefix + self._captions_ids_filename)
        captions_dict_path = os.path.join(settings.MSCOCO_DIR, prefix + self._captions_dict_filename)
        if all([os.path.isfile(images_path), os.path.isfile(captions_ids_path), os.path.isfile(captions_dict_path)]):
            self._load_jpgs_and_captions_npy()

    def _read_jpgs_and_captions(self, force_reload = False):
        # Check if 'npy' dataset already exists
        if force_reload == True:
            self._is_dataset_loaded = False
        else:
            self._try_read_npy_dataset()
            
        if self._is_dataset_loaded == False:
            images = []
            test_images = []
            captions_ids = []
            captions_dict = []

            print_info("Loading images and captions into memory...")
            with open(settings.CAPTIONS_PKL_PATH) as fd:
                cap_dict = pkl.load(fd)

            for dataset_type in ["train", "test"]:
                # Get a list of all training images full filename paths
                print_info("Loading dataset from individual JPG files and pickled dictionaries...")
                if dataset_type == "train":
                    dataset_dir = settings.TRAIN_DIR
                elif dataset_type == "test":
                    dataset_dir = settings.TEST_DIR
                log(" * {} images paths     = {}*.jpg".format(dataset_type, dataset_dir))
                images_paths = glob.glob(os.path.join(dataset_dir, "*.jpg"))
                num_images_path = len(images_paths)
                log(" * Number of {} images = {}".format(dataset_type, num_images_path))

                num_loaded_images = 0
                for i, img_path in enumerate(images_paths):
                    if dataset_type == "train" and settings.MAX_TRAINING_SAMPLES != -1:
                        if num_loaded_images >= settings.MAX_TRAINING_SAMPLES:
                            print_info("Reached maximum number of training samples: {}".format(i))
                            break
                    elif dataset_type == "test":
                        if num_loaded_images >= 1000:
                            break
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    # For now, discard greyscale images
                    if len(img_array.shape) != 3:
                        continue
                    else:
                        num_loaded_images += 1

                    if dataset_type == "train":
                        images.append(img_array)                
                        # File names look like this: COCO_train2014_000000520978.jpg
                        cap_id = os.path.basename(img_path)[:-4]
                        captions_ids.append(cap_id)
                        captions_dict.append(cap_dict[cap_id])
                    elif dataset_type == "test":
                        test_img = np.copy(img_array)
                        center = (int(np.floor(test_img.shape[0] / 2.)),
                                  int(np.floor(test_img.shape[1] / 2.)))
                        test_img[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
                        test_images.append(test_img)

                    if dataset_type == "train" and i % 5000 == 0:
                        log(" - Loaded TRAINING image #%i" % i)

                if dataset_type == "train":
                    log(" - Loaded TRAINING image #%i as the last image..." % i)
                elif dataset_type == "test":
                    log(" - Loaded TESTING image #%i as the last image..." % i)



            self.images = self.transform_images(np.array(images))
            self.test_images = self.transform_images(np.array(test_images))
            self.captions_ids = np.array(captions_ids)
            self.captions_dict = np.array(captions_dict)
            self._is_dataset_loaded = True

            log("Summary of data within dataset:")
            log(" * images.shape            = " + str(self.images.shape))
            log(" * test_images.shape       = " + str(self.test_images.shape))
            log(" * captions_ids.shape      = " + str(self.captions_ids.shape))
            log(" * captions_dict.shape     = " + str(self.captions_dict.shape))

            # Save dataset as npy file so that loading can be sped up in the future
            self._save_jpgs_and_captions_npy()

    def _load_jpgs_and_captions_npy(self):
        print_positive("Found the project datasets encoded as a 4-tensor in '.npy' format. Attempting to load...")
        prefix = ""
        if settings.MAX_TRAINING_SAMPLES != -1:
            prefix = "subset_"
        try:
            for i, filename in enumerate([self._images_filename, self._test_images_filename,  self._captions_ids_filename, self._captions_dict_filename]):
                path = os.path.join(settings.MSCOCO_DIR, prefix + filename)
                if i == 0:
                    self.images = np.load(path)
                    if settings.MAX_TRAINING_SAMPLES != -1 and self.images.shape[0] != settings.MAX_TRAINING_SAMPLES:
                        raise Exception("Incorrect number of images")
                    print_info("Loaded: {}".format(path))
                elif i == 1:
                    self.test_images = np.load(path)
                    print_info("Loaded: {}".format(path))
                elif i == 2:
                    self.captions_ids = np.load(path)
                    print_info("Loaded: {}".format(path))
                elif i == 3:
                    self.captions_dict = np.load(path)
                    print_info("Loaded: {}".format(path))
            log("")
            print_positive("Successfully loaded entire datasets!")
            self._is_dataset_loaded = True
        except Exception as e:
            handle_warning("Unable to load some of the '.npy' dataset files. Going back to loading '.jpg' files one at a time.", e)
            self._is_dataset_loaded = False
        
    def _save_jpgs_and_captions_npy(self):
        prefix = ""
        if settings.MAX_TRAINING_SAMPLES != -1:
            prefix = "subset_"
        for i, filename in enumerate([self._images_filename, self._test_images_filename, self._captions_ids_filename, self._captions_dict_filename]):
            
            path = os.path.join(settings.MSCOCO_DIR, prefix + filename)
            if i == 0:
                print_info("Writing to disk training images: {}".format(path))
                np.save(path, self.images)
            elif i == 1:
                print_info("Writing to disk test images: {}".format(path))
                np.save(path, self.test_images)
            elif i == 2:
                print_info("Writing to disk captions ids: {}".format(path))
                np.save(path, self.captions_ids)
            elif i == 3:
                print_info("Writing to disk captions dictionary: {}".format(path))
                np.save(path, self.captions_dict)

    def preprocess(self, model = settings.MODEL):
        # MUST BE IMPLEMENTED IN DERIVED CLASS
        raise NotImplemented("The function 'preprocess' MUST be implemented in the derived classes.")
                
    def normalize(self, model = settings.MODEL):
        if model == "mlp" or model == "test":
            self.images_outer_flat = normalize_data(self.images_outer_flat)
            self.images_inner_flat = normalize_data(self.images_inner_flat)
        elif model == "conv_mlp":
            self.images_outer2d = normalize_data(self.images_outer2d)
            self.images_inner_flat = normalize_data(self.images_inner_flat)
        elif model == "conv_deconv" or model == "lasagne_conv_deconv":
            self.images_outer2d = normalize_data(self.images_outer2d)
            self.images_inner2d = normalize_data(self.images_inner2d)
        elif model == "dcgan" or model == "wgan" or model == "lsgan":
            self.images = normalize_data(self.images)
            self.images_inner2d = normalize_data(self.images_inner2d)
        elif model == "vgg16":
            self.images_outer2d = self.images_outer2d.astype('float32')
            self.images_inner2d = self.images_inner2d.astype('float32')
            for col_index, subtract in enumerate([103.939, 116.779, 123.68]):
                self.images_outer2d[:,col_index,:,:] -= subtract
                self.images_inner2d[:,col_index,:,:] -= subtract
            r, g, b = self.images_outer2d[:,0,:,:], self.images_outer2d[:,1,:,:], self.images_outer2d[:,2,:,:]
            self.images_outer2d[:,0,:,:], self.images_outer2d[:,1,:,:], self.images_outer2d[:,2,:,:] = b, g, r
            r, g, b = self.images_inner2d[:,0,:,:], self.images_inner2d[:,1,:,:], self.images_inner2d[:,2,:,:]
            self.images_inner2d[:,0,:,:], self.images_inner2d[:,1,:,:], self.images_inner2d[:,2,:,:] = b, g, r
    def preload(self, test_size = 0.1, seed = 0, model = settings.MODEL):
        if model == "mlp" or model == "test":
            x = self.images_outer_flat
            y = self.images_inner_flat
            rand_seed = 1000 + seed
        elif model == "conv_mlp":
            x = self.images_outer2d
            y = self.images_inner_flat
            rand_seed = 1001 + seed
        elif model == "conv_deconv" or model == "vgg16" or model == "lasagne_conv_deconv":
            x = self.images_outer2d
            y = self.images_inner2d
            rand_seed = 1001 + seed
        elif model == "dcgan" or model == "wgan" or model == "lsgan":
            x = self.images
            y = self.images_inner2d
            rand_seed = 1002 + seed
        else:
            raise Exception("You need to specify a model for the InpaintingDataset object using 'use_model(...)'.")
            
        ### Split into training and testing data
        log("Splitting the training dataset containingg {} images into training and validation sets  after random shuffling...".format(self.images.shape[0]))
        indices = np.arange(self.images.shape[0])
        id_train, id_val = train_test_split(indices,
                                             test_size=test_size,
                                             random_state=np.random.RandomState(rand_seed))
        log("After the split, there are {} training images and {} validation images.".format(len(id_train), len(id_val)))
        
        ### Generating the training and testing datasets (80%/20% train/test split)
        #X_train, X_test, Y_train, Y_test = x[id_train], x[id_val], y[id_train], y[id_val]

        self.X = x
        self.Y = y
        self.id_train = id_train
        self.id_val = id_val

        log("Preloading is complete, with the following results:")
        log("Input training dataset X has shape     =  {0}".format(str(self.X[id_train].shape)))
        log("Output training dataset Y has shape    =  {0}".format(str(self.Y[id_train].shape)))
        log("Input validation dataset X has shape   =  {0}".format(str(self.X[id_val].shape)))
        log("Output validation dataset Y has shape  =  {0}".format(str(self.Y[id_val].shape)))

            
    def denormalize(self, model = settings.MODEL):
        if model == "mlp" or model == "test":
            self.images_outer_flat = denormalize_data(self.images_outer_flat)
            self.images_inner_flat = denormalize_data(self.images_inner_flat)
        elif model == "conv_mlp":
            self.images_outer2d = denormalize_data(self.images_outer2d)
            self.images_inner_flat = denormalize_data(self.images_inner_flat)
        elif model == "conv_deconv" or model == "vgg16" or model == "lasagne_conv_deconv":
            self.images_outer2d = denormalize_data(self.images_outer2d)
            self.images_inner2d = denormalize_data(self.images_inner2d)
        elif model == "dcgan" or model == "wgan" or model == "lsgan":
            self.images = denormalize_data(self.images)
            self.images_inner2d = denormalize_data(self.images_inner2d)


    def return_train_data(self):
        return self.X[self.id_train,], self.X[self.id_val,], \
            self.Y[self.id_train,], self.Y[self.id_val,], \
            self.id_train, self.id_val

    def return_test_data(self):
        X_test = []
        for i in range(self.test_images.shape[0]):
            X = np.copy(self.test_images[i])
            center = (int(np.floor(X.shape[1] / 2.)), int(np.floor(X.shape[2] / 2.)))
            X[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16] = 0
            X_test.append(X)
        y_test = []
        for i in range(self.test_images.shape[0]):
            y = np.copy(self.test_images[i])
            center = (int(np.floor(y.shape[1] / 2.)), int(np.floor(y.shape[2] / 2.)))
            y_test.append(y[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16])
        return normalize_data(np.array(X_test)), normalize_data(np.array(y_test))

    def get_data(self, X = False, Y = False, Train = False, Test = False):
        if X and Y:
            raise Exception("Must specify either X=True or Y=True, but not both.")
        if Train and Test:
            raise Exception("Must specify either Train=True or Test=True, but not both.")
        if X and Train:
            return self.X[self.id_train,]
        if X and Test:
            return self.X[self.id_val,]
        if Y and Train:
            return self.Y[self.id_train,]
        if Y and Test:
            return self.Y[self.id_val,]
        raise Exception("Must specify one of X or Y as True and one of Train or Test as True.")


class ImageType(object):
    images = 1 # Original 64x64 picture, untouched
    images_outer_flat = 2 # Original
    images_inner_flat = 3
    images_outer2d = 4
    images_inner2d = 5
    images_MAX = 6
    
    def __init__(self, img_type, training_dataset = True, col_channels_first = True):
        """Define a pre-processed image type. If training_dataset = False, then the image type will
        be assumed to come from the testing dataset in directory 'test2014'. If col_channels_first == False,
        then the image will be assumed to have suffix 'col_last.npy', otherwise 'col_first.npy'."""
        if img_type < 1 or image_type > ImageType.images_MAX:
            raise Exception("Invalid ImageType supplied. Valid values range between 1"
                            + "and {0}, but you supplied {1}."
                            .format(ImageType.images_MAX - 1, img_type))
        self.image_type = img_type
        self.training_dataset = training_dataset
        self.col_channels_first = col_channels_first

    def __str__(self):
        if self.training_dataset:
            filename = "train_"
        else:
            filename = "test_"
        if self.image_type == ImageType.images:
            filename += "images"
        elif self.image_type == ImageType.images_outer_flat:
            filename += "images"
        elif self.image_type == ImageType.images_inner_flat:
            filename += "images"
        elif self.image_type == ImageType.images_outer2d:
            filename += "images"
        elif self.image_type == ImageType.images_inner2d:
            filename += "images"
        if self.col_channels_first:
            filename += "_col_first.npy"
        else:
            filename += "_col_last.npy"
        return filename

class MinimalDataset(object):
    def __init__(self):
        self.dataset = {}

    def load_from_disk(self, img_type, train=True, col_first=True):
        self.image_type = ImageType(img_type, training_dataset=train, col_channels_first=col_first)
        self.image_path = os.path.join(settings.MSCOCO_DIR, str(self.image_type))
        if os.path.isfile(self.image_path):
            self.dataset[self.image_path] = np.load(self.image_path)
            return True
        return False
        
class ColorsFirstDataset(BaseDataset):
    def __init__(self, input_dim, output_dim):
        super(ColorsFirstDataset, self).__init__(input_dim, output_dim)
        self._images_filename = "train_images_col_first.npy"
        self._test_images_filename = "test_images_col_first.npy"

    def transform_images(self, images):
        """Images should be a list of numpy arrays of the form (64, 64, 3). This function will turn it into a numpy batch 4-tensor of the form (batch_size, 3, 64, 64)."""
        # Convert the list of images into a 4-tensor numpy array, then transpose the
        # image's color channel to be first: (batch_size, 3, 64, 64)
        images = np.array(images)
        images = images.transpose(0, 3, 1, 2).reshape(images.shape[0], 3, 64, 64)
        #if images.shape[1:3] != (3, 64, 64):
        #    raise ValueError("ERROR: We expected a shape of (batch_size, 3, 64, 64).")
        return images


    def preprocess(self, model = settings.MODEL):
        print_info("Preprocessing {0} images for the '{1}' model...".format(len(self.images), model))

        images_outer_flat = []
        images_inner_flat = []
        images_outer2d = []
        images_inner2d = []

        # Don't forget that here we transposed the colors channel of the original images
        # So,'img_array' will have shape (3, 64, 64).
        for i, img_array in enumerate(self.images):
            ### Get input/target from the images

            # Sanity check
            if len(img_array.shape) != 3 and img_array.shape[0] != 3:
                raise ValueError("The image #{} does not have 3 color channels.".format(i))

            ### IMPORTANT: Here width is shape[1] and height is shape[2]
            center = (int(np.floor(img_array.shape[1] / 2.)), int(np.floor(img_array.shape[2] / 2.)))

            if model == "mlp" or model == "test":
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
            elif model == "conv_mlp":
                outer_2d = np.copy(img_array)
                outer_2d[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0

                inner = np.copy(img_array)
                inner = inner[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
                inner_flat = inner.flatten()

                images_outer2d.append(outer_2d)
                images_inner_flat.append(inner_flat)
            elif model == "conv_deconv" or model == "vgg16" or model == "lasagne_conv_deconv":
                outer_2d = np.copy(img_array)
                outer_2d[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
                inner2d = np.copy(img_array)
                inner2d = inner2d[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
                images_outer2d.append(outer_2d)
                images_inner2d.append(inner2d)
            elif model == "dcgan" or model == "wgan" or model == "lsgan":
                inner2d = np.copy(img_array)
                inner2d = inner2d[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
                images_inner2d.append(inner2d)

        if model == "mlp" or model == "test":
            self.images_inner_flat = np.array(images_inner_flat)
            self.images_outer_flat = np.array(images_outer_flat)
        elif model == "conv_mlp":
            self.images_outer2d = np.array(images_outer2d)
            self.images_inner_flat = np.array(images_inner_flat)
        elif model == "conv_deconv" or model == "vgg16" or model == "lasagne_conv_deconv":
            self.images_outer2d = np.array(images_outer2d)
            self.images_inner2d = np.array(images_inner2d)
        elif model == "dcgan" or model == "wgan" or model == "lsgan":
            self.images_inner2d = np.array(images_inner2d)

class ColorsLastDataset(BaseDataset):
    def __init__(self, input_dim, output_dim):
        super(ColorsLastDataset, self).__init__(input_dim, output_dim)
        self._images_filename = "train_images_col_last.npy"
        self._test_images_filename = "test_images_col_last.npy"

    def transform_images(self, images):
        """Images should be a list of numpy arrays of the form (64, 64, 3). This function will turn it into a numpy batch 4-tensor of the form (batch_size, 64, 64, 3)."""
        # Convert the list of images into a 4-tensor numpy array
        # That's it... we're done....
        # The return value will have shape (batch_sie, 64, 64, 3)
        images = np.array(images)
        # Let's put in an assert to be sure
        #if images.shape[1:3] != (64, 64, 3):
        #    raise ValueError("ERROR: We expected a shape of (batch_size, 64, 64, 3).")
        return images


    def preprocess(self, model = settings.MODEL):
        print_info("Preprocessing {0} images for the '{1}' model...".format(len(self.images), model))

        images_outer_flat = []
        images_inner_flat = []
        images_outer2d = []
        images_inner2d = []

        # Here we didn't transpose colors channel of the original images
        # So,'img_array' will have shape (64, 64, 3).
        for i, img_array in enumerate(self.images):
            ### Get input/target from the images

            # Sanity check
            if len(img_array.shape) != 3 and img_array.shape[0] != 3:
                raise ValueError("The image #{} does not have 3 color channels.".format(i))
        
            ### IMPORTANT: Here width is shape[0] and height is shape[1]
            center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))

            if model == "mlp" or model == "test":
                outer = np.copy(img_array)
                outer_mask = np.array(np.ones(np.shape(img_array)), dtype='bool')
                outer_mask[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = False
                outer_flat = outer.flatten()
                outer_mask_flat = outer_mask.flatten()
                outer_flat = outer_flat[outer_mask_flat]

                inner = np.copy(img_array)
                inner = inner[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                inner_flat = inner.flatten()

                images_outer_flat.append(outer_flat)
                images_inner_flat.append(inner_flat)
            elif model == "conv_mlp":
                outer_2d = np.copy(img_array)
                outer_2d[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0

                inner = np.copy(img_array)
                inner = inner[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                inner_flat = inner.flatten()

                images_outer2d.append(outer_2d)
                images_inner_flat.append(inner_flat)
            elif model == "conv_deconv" or model == "vgg16" or model == "lasagne_conv_deconv":
                outer_2d = np.copy(img_array)
                outer_2d[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
                inner2d = np.copy(img_array)
                inner2d = inner2d[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                images_outer2d.append(outer_2d)
                images_inner2d.append(inner2d)
            elif model == "dcgan" or model == "wgan" or model == "lsgan":
                inner2d = np.copy(img_array)
                inner2d = inner2d[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                images_inner2d.append(inner2d)

        if model == "mlp" or model == "test":
            self.images_inner_flat = np.array(images_inner_flat)
            self.images_outer_flat = np.array(images_outer_flat)
        elif model == "conv_mlp":
            self.images_outer2d = np.array(images_outer2d)
            self.images_inner_flat = np.array(images_inner_flat)
        elif model == "conv_deconv" or model == "vgg16" or model == "lasagne_conv_deconv":
            self.images_outer2d = np.array(images_outer2d)
            self.images_inner2d = np.array(images_inner2d)
        elif model == "dcgan" or model == "wgan" or model == "lsgan":
            self.images_inner2d = np.array(images_inner2d)
