# CREDIT: This file's code was inspired by Philip Pacquette's
import os
import sys
import theano

_root = None
for path in ["/Tmp", "/tmp"]:
    if os.path.isdir(path):
        _root = path
        break

MODEL = None
EXP_NAME_PREFIX = None
EXP_NAME = None
NUM_EPOCHS = 20
BATCH_SIZE = 128

VERBOSE = 2
RELOAD_MODEL = False

theano.config.floatX = 'float32'
DATASET_AUGMENTATION = True
LOAD_BLACK_AND_WHITE_IMAGES = False
SAVE_MODEL_TO_DISK = True
SAMPLES_TO_GENERATE_PER_EPOCH = -1 # Use -1 to disable this feature

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_SAVE_DIR = _root if _root != None else BASE_DIR
MSCOCO_DIR = os.path.join(BASE_DIR, "mscoco/")
TRAIN_DIR = os.path.join(MSCOCO_DIR, "train2014/")
VALIDATE_DIR = os.path.join(MSCOCO_DIR, "validate2014/")
CAPTIONS_PKL_PATH = os.path.join(MSCOCO_DIR, "dict_key_imgID_value_caps_train_and_valid.pkl")
USER_NAME = os.environ["USER"]
LOCAL_DIR = os.path.join(ROOT_SAVE_DIR, USER_NAME, "mscoco/")
RESULTS_DIR = os.path.join(BASE_DIR, "predictions/")
SAVE_MODELS_DIR = os.path.join(BASE_DIR, "models/")

### Settings specific to the MLP model
PARAMS_MLP = { 'hidden1_neurons' : 512,
               'hidden2_neurons' : 512,
               'loss_function' : "mse",
               'optimizer' : "adam",
               'learning_rate' : "0.0002" }

PARAMS_CONV_MLP = { 'conv1_features' : 64,
                    'conv1_filter_size' : 5,
                    'maxpool1' : 2,
                    'conv2_features' : 64,
                    'conv2_filter_size' : 5,
                    'maxpool2' : 2,
                    'conv3_features' : 32,
                    'conv3_filter_size' : 5,
                    'padding' : True,
                    'hidden_last_neurons' : 256,
                    'loss_function' : "mse",
                    'optimizer' : "adam",
                    'learning_rate' : "0.0001" }

PARAMS_CONV_DECONV = { 'conv1_features' : 64,
                       'conv1_filter_size' : 5, # With no padding, output feature maps go from 64x64 input to 60x60
                       'maxpool1' : 2, # Feature maps go from 60x60 to 30x30
                       'conv2_features' : 64,
                       'conv2_filter_size' : 5, # With no padding, output feature maps go from 30x30 to 26x26
                       'maxpool2' : 2, # Feature maps go from 26x26 to 13x13
                       'conv3_features' : 32,
                       'conv3_filter_size' : 3, # With no padding, output feature maps go from 13x13 to 11x11
                       'padding' : True,
                       'deconv1_features' : 32,
                       'hidden_last_neurons' : 256,
                       'loss_function' : "mse",
                       'optimizer' : "adam",
                       'learning_rate' : "0.0001" }

