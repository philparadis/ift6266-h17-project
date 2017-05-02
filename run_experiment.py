#!/usr/bin/env python2
# coding: utf-8

import os, sys, errno, subprocess
import numpy as np
import PIL.Image as Image

# My modules
import models
import settings
from utils import normalize_data, denormalize_data
from utils import save_keras_predictions, print_results_as_html
from utils import unflatten_to_4tensor, unflatten_to_3tensor, transpose_colors_channel
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log

#######################
# Helper functions
#######################

def check_mscoco_dir():
    try:
        os.stat(settings.MSCOCO_DIR)
    except OSError, e:
        if e.errno == errno.ENOENT:
            return False
        else:
            raise e
    return True


def download_dataset():
    dataset_script="download-project-datasets.sh"
    try:
        return subprocess.call("./" + dataset_script)
    except OSError, e:
        pass
    raise OSError("Could not find the script '%s'." % dataset_script)


#################################################
# Run experiments here
# Define your global options and experiment name
# Then run the desired model
#################################################

### The experiment name is very important.

## Your model will be saved in:                           models/<experiment_name>.hdf5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

## Your predictions will be saved in: predictions/assets/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/assets/<experiment_name>/Y_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_pred_<i>.jpg


def run_experiment():
    log("Welcome! This is my final project for the course:")
    log('    IFT6266-H2017 (a.k.a. "Deep Learning"')
    log("         Prof. Aaron Courville")
    log("")
    log("This program is copyrighted 2017 Philippe Paradis. All Rights Reserved.")
    log("")
    log("Enjoy!")
    log("")
    model = None

    # Define model's specific settings
    if settings.MODEL == "test":
        model = models.Test_Model(settings.MODEL)
    elif settings.MODEL == "mlp":
        model = models.MLP_Model(settings.MODEL)
    elif settings.MODEL == "conv_mlp":
        model = models.Conv_MLP(settings.MODEL)
    elif settings.MODEL == "conv_deconv":
        model = models.Conv_Deconv(settings.MODEL)
    elif settings.MODEL == "dcgan":
        model = models.DCGAN_Model(settings.MODEL)
    elif settings.MODEL == "wgan":
        model = models.WGAN_Model(settings.MODEL)
    elif settings.MODEL == "lsgan":
        from lsgan import LSGAN_Model
        model = LSGAN_Model(settings.MODEL)
    else:
        raise NotImplementedError()

    ### Check if --force flag was passed
    if settings.FORCE_RUN:
        stopfile = os.path.join(settings.BASE_DIR, "STOP")
        if os.path.isfile(stopfile):
            os.remove(stopfile)

    ### Check for STOP file in BASE_DIR. Who knows, this experiment could
    ### be a baddy which we certainly don't want to waste precious GPU time on! Oh no!
    if model.check_stop_file():
        print_error("Oh dear, it looks like a STOP file is present in this experiment's base directory, located here:\n{}\nIf you think the STOP file was added by error and you would like to pursue this experiment further, simply feel absolute free to delete this file (which is empty, anyway).".format(os.path.join(settings.BASE_DIR, "STOP")))
        sys.exit(-2)

    ### Load checkpoint (if any). This will also load the hyper parameters file.
    ### This will also load the model's architecture, weights, optimizer states,
    ### that is, everything necessary to resume training.
    print_info("Checking for a valid checkpoint. If so, load hyper parameters and all data from the last known state...")
    checkpoint, hyperparams, resume_from_checkpoint = model.resume_last_checkpoint()
    
    if resume_from_checkpoint:
        print_positive("Found checkpoint, hyper parameters and model data all passing the integrity tests!"
                       "Ready to resume training!")
        log("")
        print_info("State of last checkpoint:")
        for key in checkpoint:
            log(" * {0: <20} = {1}".format(str(key), str(checkpoint[key])))
        log("")
    else:
        print_info("No valid checkpoint found for this experiment. Building and training model from scratch.")
        ### Build model's architecture
        model.build()
        ### Save hyperparameters to a file
        model.save_hyperparams()

    if settings.NUM_EPOCHS == 0 and not settings.PERFORM_PREDICT_ONLY:
        log("Okay, we specified 0 epochs, so we only created the experiment directory:\m{}\mand the hyper parameters file within that directory 'hyperparameters.json'.".format(settings.BASE_DIR))
        sys.exit(0)

    ###
    ### Make sure the dataset has been downloaded and extracted correctly on disk
    ###
    if check_mscoco_dir() == False:
        log("(!) The project dataset based on MSCOCO was not found in its expected location '{}' or the symlink is broken."
              .format(settings.MSCOCO_DIR))
        log("Attempting to download the dataset...")
        rc = download_dataset()
        if rc != 0:
            log("(!) Failed to download the project dataset, exiting...")
            sys.exit(rc)

    verbosity_level = "Low"
    if settings.VERBOSE == 1:
        verbosity_level = "High"
    elif settings.VERBOSE == 2:
        verbosity_level = "Medium"

    # Print info about our settings
    log("============================================================")
    print_info("Experiment name    = %s" % settings.EXP_NAME)
    log("============================================================")
    log("")
    print_info("Experiment settings and options:")
    log(" * Model type            = " + str(settings.MODEL))
    log(" * Training epochs       = " + str(settings.NUM_EPOCHS))
    log(" * Batch size            = " + str(settings.BATCH_SIZE))
    log(" * Learning rate         = " + str(settings.LEARNING_RATE))
    log(" * Epochs per checkpoint = " + str(settings.EPOCHS_PER_CHECKPOINT))
    log(" * Epochs per samples    = " + str(settings.EPOCHS_PER_SAMPLES))
    log(" * Feature Matching Loss = " + str(settings.FEATURE_MATCHING))
    log(" * Keep model's data for every checkpoint  = " + str(settings.KEEP_ALL_CHECKPOINTS))
    log(" * Verbosity             = " + str(settings.VERBOSE) + " ({})".format(verbosity_level))
    log(" * Data augmentation     = " + str(settings.DATASET_AUGMENTATION))
    log(" * Load greyscale images = " + str(settings.LOAD_BLACK_AND_WHITE_IMAGES))
    log("")

    if settings.MODEL in ["dcgan", "wgan", "lsgan"]:
        print_info("GAN-specific settings:")
        log(" * Type of GAN used      = " + str(settings.MODEL))
        log(" * Generator/critic updates per epoch = " + str(settings.UPDATES_PER_EPOCH))
        log(" * GAN learning rate     = " + str(settings.GAN_LEARNING_RATE))
        if settings.MODEL == "lsgan":
            log(" * LSGAN architecture #  = " + str(settings.LSGAN_ARCHITECTURE))
        log("")
 
    ### Print hyperparameters, as loaded from existing file or as initialized for new experiment
    print_info("Hyper parameters:")
    for key in model.hyper:
        log(" * {0: <20} = {1}".format(str(key), str(model.hyper[key])))
    log("")        

      #######################################
    # Info about the dataset
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

    import dataset

    ### Create and initialize an empty InpaintingDataset object
    Dataset = dataset.ColorsFirstDataset(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)

    ### Load dataset
    Dataset.load_dataset()

    log("")
    print_info("Summary of data within dataset:")
    log(" * images.shape        = " + str(Dataset.images.shape))
    log(" * captions_ids.shape  = " + str(Dataset.captions_ids.shape))
    log(" * captions_dict.shape = " + str(Dataset.captions_dict.shape))
    log("")

   ### Train the model (computation intensive)
    if settings.MODEL == "mlp" or settings.MODEL == "test" or settings.MODEL == "conv_mlp":
        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        model.train(Dataset)
        Dataset.denormalize()

        ### Produce predictions
        Y_test_pred = model.predict(Dataset.get_data(X=True, Test=True), batch_size = model.hyper['batch_size'])

        ### Reshape predictions to a 2d image and denormalize data
        Y_test_pred = dataset.denormalize_data(Y_test_pred)
        num_rows = Y_test_pred.shape[0]
        Y_test_pred_2d = unflatten_to_4tensor(Y_test_pred, num_rows, 32, 32, is_colors_channel_first = True)
        Y_test_pred_2d = transpose_colors_channel(Y_test_pred_2d, from_first_to_last = True)

        ### Create dataset with colors channel last
        NewDataset = dataset.ColorsLastDataset(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
        NewDataset.load_dataset()
        NewDataset.preprocess(model = "conv_deconv")
        NewDataset.preload(model = "conv_deconv")

        ### Save predictions to disk
        save_keras_predictions(Y_test_pred_2d, Dataset.id_test, NewDataset, num_images=50)
        print_results_as_html(Y_test_pred_2d, num_images=50)
    elif settings.MODEL == "conv_deconv":
        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        model.train(Dataset)
        Dataset.denormalize()

        ### Produce predictions
        Y_test_pred_2d = model.predict(Dataset.get_data(X=True, Test=True), batch_size = model.hyper['batch_size'])

        ### Reshape predictions
        Y_test_pred_2d = dataset.denormalize_data(Y_test_pred_2d)

        ### Create dataset with colors channel last
        NewDataset = dataset.ColorsLastDataset(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
        NewDataset.load_dataset()
        NewDataset.preprocess(model = "conv_deconv")
        NewDataset.preload(model = "conv_deconv")

        ### Save predictions to disk
        save_keras_predictions(Y_test_pred_2d, Dataset.id_test, NewDataset, num_images=50)
        print_results_as_html(Y_test_pred_2d, num_images=50)
    elif settings.MODEL == "dcgan":
        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        generator, discriminator, train_fn, gen_fn = model.train(Dataset, num_epochs = settings.NUM_EPOCHS, epochsize = 10, batchsize = 64, initial_eta = 8e-5)
        Dataset.denormalize()
        
        settings.touch_dir(settings.SAMPLES_DIR)
        for i in range(100):
            samples = gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100)))
            path = os.path.join(settings.EPOCHS_DIR, 'samples_%i.png' % i)
            samples = dataset.denormalize_data(samples)
            Image.fromarray(samples.reshape(10, 10, 3, 64, 64)
                            .transpose(0, 3, 1, 4, 2)
                            .reshape(10*64, 10*64, 3)).save(path)
            sample = gen_fn(lasagne.utils.floatX(np.random.rand(1, 100)))
            sample = dataset.denormalize_data(sample)
            path = os.path.join(settings.SAMPLES_DIR, 'one_sample_%i.png' % i)
            Image.fromarray(sample.reshape(3, 64, 64).transpose(1, 2, 0).reshape(64, 64, 3)).save(path)
    elif settings.MODEL == "wgan": 
        import wgan

        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        generator, critic, generator_train_fn, critic_train_fn, gen_fn = wgan.train(Dataset, num_epochs=settings.NUM_EPOCHS)
        Dataset.denormalize()
        
        settings.touch_dir(settings.SAMPLES_DIR)
        for i in range(100):
            samples = gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100)))
            path = os.path.join(settings.EPOCHS_DIR, 'samples_%i.png' % i)
            samples = dataset.denormalize_data(samples)
            Image.fromarray(samples.reshape(10, 10, 3, 64, 64)
                            .transpose(0, 3, 1, 4, 2)
                            .reshape(10*64, 10*64, 3)).save(path)
            sample = gen_fn(lasagne.utils.floatX(np.random.rand(1, 100)))
            sample = dataset.denormalize_data(sample)
            path = os.path.join(settings.SAMPLES_DIR, 'one_sample_%i.png' % i)
            Image.fromarray(sample.reshape(3, 64, 64).transpose(1, 2, 0).reshape(64, 64, 3)).save(path) 
    elif settings.MODEL == "lsgan": 
        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        
        generator, critic, gen_fn = model.train(Dataset, num_epochs = settings.NUM_EPOCHS,
                                                epochsize = settings.UPDATES_PER_EPOCH,
                                                batchsize = settings.BATCH_SIZE,
                                                architecture = settings.LSGAN_ARCHITECTURE,
                                                initial_eta = settings.GAN_LEARNING_RATE)

        Dataset.denormalize()
        
        settings.touch_dir(settings.SAMPLES_DIR)
        for i in range(100):
            samples = gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100)))
            path = os.path.join(settings.EPOCHS_DIR, 'samples_%i.png' % i)
            samples = dataset.denormalize_data(samples)
            Image.fromarray(samples.reshape(10, 10, 3, 64, 64)
                            .transpose(0, 3, 1, 4, 2)
                            .reshape(10*64, 10*64, 3)).save(path)
            sample = gen_fn(lasagne.utils.floatX(np.random.rand(1, 100)))
            sample = dataset.denormalize_data(sample)
            path = os.path.join(settings.SAMPLES_DIR, 'one_sample_%i.png' % i)
            Image.fromarray(sample.reshape(3, 64, 64).transpose(1, 2, 0).reshape(64, 64, 3)).save(path)
    elif settings.MODEL == "conv_deconv":
        pass
    

    ### Success...? Well, at least we didn't crash :P
    log("Exiting normally. That's typically a good sign :-)")
    sys.exit(0)

