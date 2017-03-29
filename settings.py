# CREDIT: This file's code was inspired by Philip Pacquette's
import os
import theano

class Settings:
    MODEL = None
    EXP_NAME = None
    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    VERBOSE = 2
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    MSCOCO_DIR = os.path.join(ROOT_DIR, "mscoco/")
    TRAIN_DIR = os.path.join(MSCOCO_DIR, "train2014/")
    VALIDATE_DIR = os.path.join(MSCOCO_DIR, "validate2014/")
    CAPTIONS_PKL_PATH = os.path.join(MSCOCO_DIR, "dict_key_imgID_value_caps_train_and_valid.pkl")
    USER_NAME = os.environ["USER"]
    LOCAL_DIR = os.path.joint("/Tmp", USER_NAME, "mscoco/")
    RESULTS_DIR = os.path.join(ROOT_DIR, "predictions/")
    SAVE_MODELS_DIR = os.path.join(ROOT_DIR, "models/")
    theano.config.floatX = 'float32'
    DATASET_AUGMENTATION = True
    LOAD_BLACK_AND_WHITE_IMAGES = False
    SAVE_MODEL_TO_DISK = True
    SAMPLES_TO_GENERATE_PER_EPOCH = -1 # Use -1 to disable this feature

    @staticmethod
    def get_attributes():
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print members

    @staticmethod
    def validate_parameters():
        members = Settings.get_attributes()
        for attr in members:
            if attr == None:
                return Exception()
