#!/usr/bin/env python2
# coding: utf-8

import os, sys
import argparse
import datetime
from IPython.core import ultratb

def initialize_directories():
    import settings
    settings.BASE_DIR    = os.path.join(settings.MODEL, settings.EXP_NAME)
    settings.MODELS_DIR  = os.path.join(settings.BASE_DIR, "models/")
    settings.EPOCHS_DIR  = os.path.join(settings.BASE_DIR, "epochs/")
    settings.PERF_DIR    = os.path.join(settings.BASE_DIR, "performance/")
    settings.SAMPLES_DIR = os.path.join(settings.BASE_DIR, "samples/")
    settings.PRED_DIR    = os.path.join(settings.BASE_DIR, "predictions/")
    settings.ASSETS_DIR  = os.path.join(settings.PRED_DIR, "assets/")
    settings.HTML_DIR    = settings.PRED_DIR
    settings.CHECKPOINTS_DIR = os.path.join(settings.BASE_DIR, "checkpoints/")
    settings.LOGS_DIR    = os.path.join(settings.BASE_DIR, "logs/")

    # Create the directories
    settings.touch_dir(settings.BASE_DIR)
    settings.touch_dir(settings.MODELS_DIR)
    settings.touch_dir(settings.EPOCHS_DIR)
    settings.touch_dir(settings.PERF_DIR)
    settings.touch_dir(settings.SAMPLES_DIR)
    settings.touch_dir(settings.PRED_DIR)
    settings.touch_dir(settings.ASSETS_DIR)
    settings.touch_dir(settings.HTML_DIR)
    settings.touch_dir(settings.CHECKPOINTS_DIR)
    settings.touch_dir(settings.LOGS_DIR)

    # Set some file paths
    settings.OUTLOGFILE = os.path.join(settings.LOGS_DIR, "output.log")
    settings.ERRLOGFILE = os.path.join(settings.LOGS_DIR, "errors.log")

    # Clean up the log files if they already are present (rather than append to them, 
    if os.path.isfile(settings.OUTLOGFILE):
        os.remove(settings.OUTLOGFILE)
    if os.path.isfile(settings.ERRLOGFILE):
        os.remove(settings.ERRLOGFILE)

if __name__ == "__main__":
    import settings
    import numpy as np

    np.random.seed(0)

    ### Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model",
                        help="Model type, among: mlp, conv_mlp, conv_deconv, lasagne_conv_deconv, lasagne_conv_deconv_dropout, vgg16, dcgan, wgan, lsgan.")
    parser.add_argument("exp_name_prefix", help="Prefix used at the beginning of the name of the experiment. Your results will be stored in various subfolders and files which start with this prefix. The exact name of the experiment depends on the model used and various hyperparameters.")
    parser.add_argument("-v", "--verbose", type=int,
                        default=settings.VERBOSE, help="0 means quiet, 1 means verbose and 2 means limited verbosity.")
    parser.add_argument("-e", "--epochs", type=int,
                        default=settings.NUM_EPOCHS,
                        help="Number of epochs to train (either for a new model or *extra* epochs when resuming an experiment.")
#    parser.add_argument("-i", "--init", action="store_true", help="Only initialize the experiment directory with the hyperparams.json file, without actually running any computations (helpful to tweak manually the hyperparameters)")
#    parser.add_argument("-m", "--max_epochs", type=int,
#                        default=settings.MAX_EPOCHS,
#                        help="Maximum number of epochs to train (useful in case training gets interrupted, to avoid overtraining a model.")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=settings.BATCH_SIZE, help="Only use this if you want to override the default hyper parameter value for your model. It may be better to create an experiment directory with an 'hyperparameters.json' file and tweak the parameters there.")
    parser.add_argument("-l", "--learning_rate", type=float, default=settings.LEARNING_RATE,
                        help="Only set this if you want to override the default hyper parameter value for your model. It may be better to create an experiment directory with an 'hyperparameters.json' file and tweak the parameters there.")
    parser.add_argument("-g", "--gan_learning_rate", type=float, default=settings.GAN_LEARNING_RATE,
                        help="Only set this if you want to override the default hyper parameter value for your GAN model. It may be better to create an experiment directory with an 'hyperparameters.json' file and tweak the parameters there.") 
    parser.add_argument("-f", "--force", action="store_true", default=settings.FORCE_RUN,
                        help="Force the experiment to run even if a STOP file is present. This will also delete the STOP file.")
#    parser.add_argument("-o", "--optimizer", type=string,
#                        default=settings.OPTIMIZER, help="Optimizer (available: adam, sgd, rmsprop, adamax, nadam)")
    parser.add_argument("-c", "--epochs_per_checkpoint", type=int, default=settings.EPOCHS_PER_CHECKPOINT,
                        help="Amount of epochs to perform during training between every checkpoint.")
    parser.add_argument("-s", "--epochs_per_samples", type=int, default=settings.EPOCHS_PER_SAMPLES,
                        help="Amount of epochs to perform during training between every generation of image samples (typically 100 images in a 10x10 grid). If your epochs are rather short, you might want to increase this value, as generating images and saving them to disk can be relatively costly.")
    parser.add_argument("-m", "--max_training_samples", type=int, default=settings.MAX_TRAINING_SAMPLES,
                        help="Maximum number of training samples to use. Should be between 1000 and 82611 (or equivalently, None, to use the entire training dataset.")
    parser.add_argument("-r", "--ratio", type=float, default=settings.RATIO_VGG_LOSS,
                        help="Ratio between the target-prediction L2 loss and the target-prediction loss as output of a VGG-16 intermediate convolutional layer (layer 'conv4_2' by default). Typical values may be 0.001, but you can increase it such as 0.1 to favor the VGG feature maps space loss or decrease it such as 0.00001 to favor the pixel space L2 loss.")
    parser.add_argument("-k", "--keep_all_checkpoints", action="store_true", default=settings.KEEP_ALL_CHECKPOINTS, help="By default, only the model saved during the last checkpoint is saved. Pass this flag if you want to keep a models on disk with its associated epoch in the filename at every checkpoint.")
    parser.add_argument("-a", "--architecture", type=int, default=settings.LSGAN_ARCHITECTURE,
                        help="Architecture type, only applies to the LSGAN critic's neural network (values: 0, 1 or 2).")
    parser.add_argument("-u", "--updates_per_epoch", type=int, default=settings.UPDATES_PER_EPOCH,
                        help="Number of times to update the generator and discriminator/critic per epoch. Applies to GAN models only.")
#    parser.add_argument("-m", "--feature_matching", action="store_true", default=settings.FEATURE_MATCHING, help="By default, feature matching is not used (equivalently, it is set to 0, meaning that the loss function uses the last layer's output). You can set this value to 1 to use the output of the second-to-last layer, or a value of 2 to use the output of the third-to-last layer, and so on. This technique is called 'feature matching' and many provide benefits in some cases. Note that it is not currently implemented in all models and you will receive a message indicating if feature matching is used for your model.")
    parser.add_argument("-t", "--tiny", action="store_true", default=settings.TINY_DATASET, help="Use a tiny dataset containing only 5000 training samples and 500 test samples, for testing purposes.")
#                        help="Use CPU instead of GPU. Used for debugging and testing purposes.")
    parser.add_argument("-d", "--debug", action="store_true", default=settings.DEBUG_MODE, help="Enable debug mode. This will hook the debugger to the program's exceptions; that is, whenever an unhandled exception is raised, pdb will be launched to examine the program post-mortem, instead of the program automatically exiting.")

    args = parser.parse_args()
    settings.MODEL = args.model.lower()
    settings.EXP_NAME_PREFIX = args.exp_name_prefix
    settings.VERBOSE = args.verbose
#    settings.PERFORM_INIT_ONLY = args.init
    settings.NUM_EPOCHS = args.epochs
    settings.BATCH_SIZE = args.batch_size
    settings.LEARNING_RATE = args.learning_rate
    settings.GAN_LEARNING_RATE = args.gan_learning_rate
    settings.FORCE_RUN = args.force
    settings.EPOCHS_PER_CHECKPOINT = args.epochs_per_checkpoint
    settings.EPOCHS_PER_SAMPLES = args.epochs_per_samples
    settings.MAX_TRAINING_SAMPLES = args.max_training_samples
    settings.RATIO_VGG_LOSS = args.ratio
    settings.KEEP_ALL_CHECKPOINTS = args.keep_all_checkpoints
    settings.LSGAN_ARCHITECTURE = args.architecture
    settings.UPDATES_PER_EPOCH = args.updates_per_epoch
    #settings.FEATURE_MATCHING = args.feature_matching
    settings.TINY_DATASET = args.tiny
    #settings.USE_CPU = args.cpu
    settings.DEBUG_MODE = args.debug

    if settings.DEBUG_MODE:
        sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)        

    if not settings.MODEL in ["test", "mlp", "conv_mlp", "conv_deconv", "lasagne_conv_deconv", "lasagne_conv_deconv_dropout", "vgg16", "dcgan", "wgan", "lsgan"]:
        raise NotImplementedError("The model '{}' is not yet implemented yet, sorry!".format(settings.MODEL))

    from utils import print_warning, handle_error, log

    ### Setup logging and xtraceback
    try:
        import logging
        #try:
            #import xtraceback
            #xtraceback.compat.install(options={'print_width':60})
            #settings.MODULE_HAVE_XTRACEBACK = True
        #except ImportError as e:
        #    print_warning("You do not have the 'xtraceback' module. " +
        #                  "Please consider installing it, and all other " +
        #                  "required or optional modules, via the command " +
        #                  "line \"> pip install -r requirements.txt\"")
    except ImportError as e:
        print_warning("You do not have the 'logging' module. Please consider" +
                      " installing it, and all other required or optional " +
                      "modules, via the command line \"> pip install -r requirements.txt\"")

    ### Initialize experiment directories and global variables
    ## Set the experiment name according to a boring structure (but simple is good :D)
    settings.EXP_NAME = "{}_model_{}".format(settings.EXP_NAME_PREFIX, settings.MODEL)

    ## Initialize global variables that store the various directories where
    ## results will be saved for this experiment. Moreover, create them.
    initialize_directories()

    ### All systems are go, let's fire up this ship to infinity and beyond!
    from run_experiment import run_experiment
    t = datetime.datetime.now()
    log("")
    log("============================================================")
    log("Current stardate: {0}".format(t.strftime("%Y-%m-%d %H:%M:%S")))
    log("")
    log("All output is logged on disk to: {}".format(settings.OUTLOGFILE))
    log("All errors are logged on disk to: {}".format(settings.ERRLOGFILE))
    log("")

    run_experiment()

    ### Graceful exit
    sys.exit(0)
