import os

_root = None
for path in ["/Tmp", "/tmp"]:
    if os.path.isdir(path):
        _root = path
        break

# Command line arguments
MODEL = None
EXP_NAME_PREFIX = None
EXP_NAME = None
NUM_EPOCHS = 20
MAX_EPOCHS = 2000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
GAN_LEARNING_RATE = 1e-4
FORCE_RUN = False
EPOCHS_PER_CHECKPOINT = 20
EPOCHS_PER_SAMPLES = 1
KEEP_ALL_CHECKPOINTS = False
VERBOSE = 2
RELOAD_MODEL = False
USE_CPU = False
PERFORM_INIT_ONLY = False
PERFORM_PREDICT_ONLY = False
LOSS_FUNCTION = "mse"
OPTIMIZER = "adam"
LSGAN_ARCHITECTURE = 1
UPDATES_PER_EPOCH = 10
FEATURE_MATCHING = False
TRAINING_BATCH_SIZE = 0 # Total number of training examples (i.e. size of first dimension of training tensor)

THEANO_CONFIG_FLOATX = 'float32'
DATASET_AUGMENTATION = True
LOAD_BLACK_AND_WHITE_IMAGES = False
SAVE_MODEL_TO_DISK = True
SAMPLES_TO_GENERATE_PER_EPOCH = -1 # Use -1 to disable this feature

# Modules available
MODULE_HAVE_XTRACEBACK = False

# Dataset settings
IMAGE_WIDTH  = 64
IMAGE_HEIGHT = 64


# Datasets directories
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
MSCOCO_DIR = os.path.join(THIS_DIR, "mscoco/")
TRAIN_DIR = os.path.join(MSCOCO_DIR, "train2014/")
VALIDATE_DIR = os.path.join(MSCOCO_DIR, "validate2014/")
CAPTIONS_PKL_PATH = os.path.join(MSCOCO_DIR, "dict_key_imgID_value_caps_train_and_valid.pkl")

# Experiment results directories
BASE_DIR = None # Will be defined at runtime as: os.path.join(MODEL, EXP_NAME)
MODELS_DIR = None # Location to save trained models
EPOCHS_DIR = None # Location to save epochs by epochs samples, results or other evolving data
PERF_DIR = None # Location for performance results (training/testing loss, timing, etc.)
SAMPLES_DIR = None # Location for random samples (for generative models)
PRED_DIR = None # Location for model predictions
ASSETS_DIR = None # Location to store the various images produced for HTML visualization
HTML_DIR = None # Location to store the HTML page(s) for nicer visualization of results
CHECKPOINTS_DIR = None # Location to store the various checkpoints files
LOGS_DIR = None # Location for the log files produced during program execution, errors, stack traces,

OUTLOGFILE = None
ERRLOGFILE = None

def touch_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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

