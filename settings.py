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
EXP_NAME = None
NUM_EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 2
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
theano.config.floatX = 'float32'
DATASET_AUGMENTATION = True
LOAD_BLACK_AND_WHITE_IMAGES = False
SAVE_MODEL_TO_DISK = True
SAMPLES_TO_GENERATE_PER_EPOCH = -1 # Use -1 to disable this feature
