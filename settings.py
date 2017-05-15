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
NUM_EPOCHS = 30
MAX_TRAINING_SAMPLES = None
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAN_LEARNING_RATE = 0.001
FORCE_RUN = False
EPOCHS_PER_CHECKPOINT = 5
EPOCHS_PER_SAMPLES = 5
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
TINY_DATASET = False

ACTUAL_TRAINING_SAMPLES = 82611

USE_VGG16_LOSS = False

THEANO_CONFIG_FLOATX = 'float32'
DATASET_AUGMENTATION = False
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
TEST_DIR = os.path.join(MSCOCO_DIR, "test2014/")
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

