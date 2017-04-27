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
from utils import print_critical, print_error, print_warning, print_info, print_positive

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


def initialize_directories():
    settings.BASE_DIR    = os.path.join(settings.MODEL, settings.EXP_NAME)
    settings.MODELS_DIR  = os.path.join(settings.BASE_DIR, "models/")
    settings.EPOCHS_DIR  = os.path.join(settings.BASE_DIR, "epochs/")
    settings.PERF_DIR    = os.path.join(settings.BASE_DIR, "performance/")
    settings.SAMPLES_DIR = os.path.join(settings.BASE_DIR, "samples/")
    settings.PRED_DIR    = os.path.join(settings.BASE_DIR, "predictions/")
    settings.ASSETS_DIR  = os.path.join(settings.PRED_DIR, "assets/")
    settings.HTML_DIR    = settings.PRED_DIR
    settings.CHECKPOINTS_DIR = os.path.join(settings.BASE_DIR, "checkpoints/")


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
    model = None
    loss_function = None
    
    # Define model's specific settings
    if settings.MODEL == "test":
        model = models.Test_Model(settings.MODEL)
    elif settings.MODEL == "mlp":
        model = models.MLP_Model(settings.MODEL)
    elif settings.MODEL == "dcgan":
        model = models.DCGAN_Model(settings.MODEL)
    elif settings.MODEL == "wgan":
        model = models.WGAN_Model(settings.MODEL)
    elif settings.MODEL == "lsgan":
        model = models.LSGAN_Model(settings.MODEL)
    else:
        raise NotImplementedError()
    
    settings.EXP_NAME = "{}_model_{}".format(settings.EXP_NAME_PREFIX, settings.MODEL)

    ## Update the hyperparameter to match the argument supplied on the command line
    model.hyper['learning_rate'] = settings.LEARNING_RATE
    model.hyper['batch_size'] = settings.BATCH_SIZE

    ### Make sure the dataset has been downloaded and extracted correctly on disk
    if check_mscoco_dir() == False:
        print("(!) The project dataset based on MSCOCO was not found in its expected location '{}' or the symlink is broken."
              .format(settings.MSCOCO_DIR))
        print("Attempting to download the dataset...")
        rc = download_dataset()
        if rc != 0:
            print("(!) Failed to download the project dataset, exiting...")
            sys.exit(rc)

    ### Initialize global variables that store the various directories where
    ### results will be saved for this experiment. Moreover, create them.
    initialize_directories()

    verbosity_level = "Low"
    if settings.VERBOSE == 0:
        verbosity_level = "High"
    elif settings.VERBOSE == 2:
        verbosity_level = "medium"

    # Print info about our settings
    print("IFT6266-H2017 (Prof. Aaron Courville)")
    print("Final Project")
    print("Copyright 2017 Philippe Paradis. All Rights Reserved.")
    print("")
    print("============================================================")
    print("* Experiment name  = %s" % settings.EXP_NAME)
    print("============================================================")
    print("")
    print("Settings:")
    print(" * Training epochs       = " + str(settings.NUM_EPOCHS))
    print(" * Verbosity             = " + verbosity_level)
    print(" * Dta augmentation      = " + str(settings.DATASET_AUGMENTATION))
    print(" * Load greyscale images = " + str(not settings.LOAD_BLACK_AND_WHITE_IMAGES))
    print("")
    
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
    print("Finished loading dataset...")
    print("")
    
    print("Summary of data within dataset:")
    print(" * images.shape            = " + str(Dataset.images.shape))
    print(" * captions_ids.shape      = " + str(Dataset.captions_ids.shape))
    print(" * captions_dict.shape     = " + str(Dataset.captions_dict.shape))
    # print("images_outer2d.shape    = " + str(Dataset.images_outer2d.shape))
    # print("images_inner2d.shape    = " + str(Dataset.images_inner2d.shape))
    # print("images_outer_flat.shape = " + str(Dataset.images_outer_flat.shape))
    # print("images_inner_flat.shape = " + str(Dataset.images_inner_flat.shape))
    # print("images_T.shape          = " + str(Dataset.images_T.shape))
    # print("images_outer2d_T.shape  = " + str(Dataset.images_outer2d_T.shape))
    # print("images_inner2d_T.shape  = " + str(Dataset.images_inner2d_T.shape))
    print("")

    ### Load checkpoint (if any). This will also load the hyper parameters file.
    ### This will also load the model's architecture, weights, optimizer states,
    ### that is, everything necessary to resume training.
    checkpoint = model.load_checkpoint()
    if checkpoint != None:
        print_positive("Ready to resume from a valid checkpoint!")
        print("")
        print_info("State of last checkpoint:")
        for key in checkpoint:
            print(" * {0: <20} = {1}".format(str(key), str(checkpoint[key])))
        print("")
    else:
        ### Build model's architecture
        print("(!) No valid checkpoint found for this experiment. Building and training model from scratch.")
        if settings.MODEL == "mlp" or settings.MODEL == "test":
            model.build()
        elif settings.MODEL == "conv_mlp":
            model.build()
        elif settings.MODEL == "dcgan":
            pass
        elif settings.MODEL == "wgan":
            pass
        elif settings.MODEL == "lsgan":
            pass
        else:
            raise NotImplementedError()
        
        ### Save hyperparameters to a file
        model.save_hyperparams()

    ### Print hyperparameters, as loaded from existing file or as initialized for new experiment
    print_info("Hyperparameters:")
    for key in model.hyper:
        print(" * {0: <20} = {1}".format(str(key), str(model.hyper[key])))
    print("")        

    ### Train the model (computation intensive)
    if settings.MODEL == "mlp" or settings.MODEL == "test":
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
        NewDataset.preprocess(model = "conv_mlp")
        NewDataset.preload(model = "conv_mlp")

        ### Save predictions to disk
        save_keras_predictions(Y_test_pred_2d, Dataset.id_test, NewDataset, num_images=50)
        print_results_as_html(Y_test_pred_2d, num_images=50)
    elif settings.MODEL == "conv_mlp":
        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        model.train(Dataset)
        Dataset.denormalize()

        ### Produce predictions
        Y_test_pred = model.predict(Dataset.get_data(X=True, Test=True), batch_size = model.hyper['batch_size'])

        ### Reshape predictions
        Y_test_pred = dataset.denormalize_data(Y_test_pred)
        num_rows = Y_test_pred.shape[0]
        Y_test_pred_2d = unflatten_to_4tensor(Y_test_pred, num_rows, 32, 32, is_colors_channel_first = True)
        Y_test_pred_2d = transpose_colors_channel(Y_test_pred_2d, from_first_to_last = True)

        ### Save predictions to disk

        ### Create a new dataset based on original images
        ### We need 'outer2d' and 'inner2d' images even if we did not use them in the model, in order to produce our results.
        print_positive("We finished training the model and obtained predictions for the image inpainting tasks.")
        print_positive("However, we need to go back to the original images with the colors channel last.")
        print_positive("To do so, we create a new class ColorsLastDataset that will take care of loading the data in the right format.")
        NewDataset = dataset.ColorsLastDataset(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
        NewDataset.load_dataset()
        print("Summary of data within ColorsLastDataset:")
        print(" * images.shape            = " + str(NewDataset.images.shape))
        print(" * captions_ids.shape      = " + str(NewDataset.captions_ids.shape))
        print(" * captions_dict.shape     = " + str(NewDataset.captions_dict.shape))
        NewDataset.preprocess()
        NewDataset.preload(model = "conv_mlp")

        save_keras_predictions(Y_test_pred_2d, Dataset.test.id, NewDataset, num_images=50)
        print_results_as_html(Y_test_pred_2d, num_images=50)
    elif settings.MODEL == "dcgan":
        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        model.train(Dataset)
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
        import wgan_lasagne

        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        generator, critic, generator_train_fn, critic_train_fn, gen_fn = wgan_lasagne.train(Dataset, num_epochs=settings.NUM_EPOCHS)
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
        import lsgan_lasagne

        Dataset.preprocess()
        Dataset.normalize()
        Dataset.preload()
        try:
            geneqrator, critic, generator_train_fn, critic_train_fn, gen_fn = lsgan_lasagne.train(Dataset, num_epochs=settings.NUM_EPOCHS)
        except Exception, e:
            print("Error while training LS-GAN model. Adding STOP file.")
            raise e
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
    print("Exiting normally. That's typically a good sign :-)")
    sys.exit(0)

