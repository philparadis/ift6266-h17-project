#!/usr/bin/env python2
# coding: utf-8

import argparse
import numpy as np

import dataset
import settings
import models
from save_results import *

#################################################
# Run experiments here
# Define your global options and experiment name
# Then run the desired model
#################################################

### The experiment name is very important.

## Your model will be saved in:                           models/<experiment_name>.h5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

## Your predictions will be saved in: predictions/assets/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/assets/<experiment_name>/Y_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_pred_<i>.jpg

def run():
    model = None
    loss_function = None
    
    # Define model's specific settings
    if settings.MODEL == "mlp":
        input_dim = 64*64*3 - 32*32*3
        output_dim = 32*32*3
        loss_function = "mse"
        model_params = models.ModelParameters(settings.MODEL, input_dim, output_dim, loss_function)
        model = models.build_mlp(model_params, input_dim, output_dim)
    else:
        raise NotImplementedError()
    
    settings.EXP_NAME = "exp_model-%s_loss-%s_epochs-%i" \
                        % (settings.MODEL, loss_function, settings.NUM_EPOCHS)

    # Print info about our settings
    print("============================================================")
    print("Experiment name = %s" % settings.EXP_NAME)
    print("============================================================")
    print("Hyperparameters:")
    print("------------------------------------------------------------")
    print(" * Model         = " + settings.MODEL)
    print(" * Epochs        = " + str(settings.NUM_EPOCHS))
    print(" * Batch size    = " + str(settings.BATCH_SIZE))
    print("============================================================")
    print("Other settings:")
    print("------------------------------------------------------------")
    print(" * Using data augmentation         = " + str(settings.DATASET_AUGMENTATION))
    print(" * Loading black and white images  = " + str(settings.LOAD_BLACK_AND_WHITE_IMAGES))
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

    ### Create and initialize an empty InpaintingDataset object
    Dataset = dataset.InpaintingDataset(input_dim, output_dim)

    ### Load dataset
    Dataset.read_jpgs_and_captions_and_flatten()

    print("Finished loading and pre-processing datasets...")
    print("Summary of datasets:")
    print("images.shape            = " + str(Dataset.images.shape))
    print("images_outer2d.shape    = " + str(Dataset.images_outer2d.shape))
    print("images_inner2d.shape    = " + str(Dataset.images_inner2d.shape))
    print("images_outer_flat.shape = " + str(Dataset.images_outer_flat.shape))
    print("images_inner_flat.shape = " + str(Dataset.images_inner_flat.shape))
    print("captions_ids.shape      = " + str(Dataset.captions_ids.shape))
    print("captions_dict.shape     = " + str(Dataset.captions_dict.shape))

     ###
    if settings.MODEL == "mlp":
        model = models.train_mlp(model, model_params, Dataset)
    elif settings.MODEL == "dcgan":
        model = models.train_dcgan(Dataset)
    
    ### Produce predictions
    Y_test_pred = model.predict(X_test, batch_size=settings.BATCH_SIZE)

    # Reshape predictions to a 2d image and denormalize data
    Y_test_pred = denormalize_data(Y_test_pred)
    num_rows = Y_test_pred.shape[0]
    Y_test_pred_2d = np.reshape(Y_test_pred, (num_rows, 32, 32, 3))

    ### Save predictions to disk
    save_performance_results(model, X_train, Y_train, X_test, Y_test)
    save_predictions_info(Y_test_pred_2d, id_test, Dataset, num_images=50)
    print_results_as_html(Y_test_pred_2d, num_images=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model choice (current options: mlp, convnet, convnet_lstm, vae, dcgan)")
    parser.add_argument("experiment_name", help="Name of experiment. Your results will be stored in subfolders with this name.")
    parser.add_argument("-v", "--verbose", type=int,
                        default=settings.VERBOSE, help="0 means quiet, 1 means verbose and 2 means limited verbosity.")
    parser.add_argument("-e", "--num_epochs", type=int,
                        default=settings.NUM_EPOCHS, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=settings.BATCH_SIZE, help="Size of minibatches")
    parser.add_argument("-l", "--load_model_from_file", default=None,
                        help="Load HF5 model from subdirectory 'models'. This will skip the training phase.")

    args = parser.parse_args()
    settings.MODEL = args.model
    settings.EXPERIMENT_NAME = args.experiment_name
    settings.NUM_EPOCHS = args.num_epochs
    settings.BATCH_SIZE = args.batch_size
    settings.VERBOSE = args.verbose

    if settings.MODEL in ["convnet", "convnet_lstm", "vae", "dcgan"]:
        raise NotImplementedError("The model '{}' is not yet implemented yet, sorry!".format(settings.MODEL))
    
    run()
