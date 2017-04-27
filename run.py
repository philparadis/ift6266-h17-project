#!/usr/bin/env python2
# coding: utf-8

import logging
import xtraceback
import argparse
import settings

from run_experiment import run_experiment

if __name__ == "__main__":
    ### Setup xtraceback
    xtraceback.compat.install()

    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Model choice (current options: test, mlp, conv_mlp*, conv_lstm*, vae*, conv_autoencoder*, dcgan, wgan, lsgan (*: Models with * may not be fully implemented yet).)")
    parser.add_argument("exp_name_prefix", help="Prefix used at the beginning of the name of the experiment. Your results will be stored in various subfolders and files which start with this prefix. The exact name of the experiment depends on the model used and various hyperparameters.")
    parser.add_argument("-v", "--verbose", type=int,
                        default=settings.VERBOSE, help="0 means quiet, 1 means verbose and 2 means limited verbosity.")
    parser.add_argument("-e", "--epochs", type=int,
                        default=settings.NUM_EPOCHS,
                        help="Number of epochs to train (either for a new model or extra epochs when resuming an existing model.")
    parser.add_argument("-m", "--max_epochs", type=int,
                        default=settings.MAX_EPOCHS,
                        help="Maximum number of epochs to train (useful in case training gets interrupted, to avoid overtraining a model.")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=settings.BATCH_SIZE, help="Size of minibatches.")
    parser.add_argument("-l", "--learning_rate", type=float, default=settings.LEARNING_RATE,
                        help="Learning rate of the optimizer (may be better to leave empty is you're not sure what you're doing, the default value of None will select a learning rate appropriate for the model type).")
#    parser.add_argument("-f", "--loss_function", type=string,
#                        default=settings.LOSS_FUNCTION, help="Loss function (available: mse, mae, categorical_crossentropy, binary_crossentropy, kullback_leibler_divergence, hinge)")
#    parser.add_argument("-o", "--optimizer", type=string,
#                        default=settings.OPTIMIZER, help="Optimizer (available: adam, sgd, rmsprop, adamax, nadam)")
    parser.add_argument("-c", "--epochs_per_checkpoint", type=int, default=settings.EPOCHS_PER_CHECKPOINT,
                        help="Amount of epochs to perform during training between every checkpoint.")
    parser.add_argument("--cpu", action="store_true", default=settings.USE_CPU,
                        help="Use CPU instead of GPU. Used for debugging and testing purposes.")

    args = parser.parse_args()
    settings.MODEL = args.model.lower()
    settings.EXP_NAME_PREFIX = args.exp_name_prefix
    settings.VERBOSE = args.verbose
    settings.NUM_EPOCHS = args.epochs
    settings.MAX_EPOCHS = args.max_epochs
    settings.BATCH_SIZE = args.batch_size
    settings.LEARNING_RATE = args.learning_rate
    settings.EPOCHS_PER_CHECKPOINT = args.epochs_per_checkpoint
    settings.USE_CPU = args.cpu

    if not settings.MODEL in ["test", "mlp", "conv_mlp", "dcgan", "wgan", "lsgan"]:
        raise NotImplementedError("The model '{}' is not yet implemented yet, sorry!".format(settings.MODEL))
    
    run_experiment()
