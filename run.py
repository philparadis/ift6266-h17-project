#!/usr/bin/env python2
# coding: utf-8

import argparse
import settings


if __name__ == "__main__":
        
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        help="Model choice (current options: test, mlp, conv_mlp*, conv_lstm*, vae*, conv_autoencoder*, dcgan, wgan, lsgan (*: Models with * may not be fully implemented yet).)")
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
    parser.add_argument("-f", "--force", action="store_true", default=settings.FORCE_RUN,
                        help="Force the experiment to run even if a STOP file is present. This will also delete the STOP file.")
#    parser.add_argument("-o", "--optimizer", type=string,
#                        default=settings.OPTIMIZER, help="Optimizer (available: adam, sgd, rmsprop, adamax, nadam)")
    parser.add_argument("-c", "--epochs_per_checkpoint", type=int, default=settings.EPOCHS_PER_CHECKPOINT,
                        help="Amount of epochs to perform during training between every checkpoint.")
#    parser.add_argument("--cpu", action="store_true", default=settings.USE_CPU,
#                        help="Use CPU instead of GPU. Used for debugging and testing purposes.")

    args = parser.parse_args()
    settings.MODEL = args.model.lower()
    settings.EXP_NAME_PREFIX = args.exp_name_prefix
    settings.VERBOSE = args.verbose
#    settings.PERFORM_INIT_ONLY = args.init
    settings.NUM_EPOCHS = args.epochs
    #settings.MAX_EPOCHS = args.max_epochs
    settings.BATCH_SIZE = args.batch_size
    settings.LEARNING_RATE = args.learning_rate
    settings.FORCE_RUN = args.force
    settings.EPOCHS_PER_CHECKPOINT = args.epochs_per_checkpoint
#    settings.USE_CPU = args.cpu

    if not settings.MODEL in ["test", "mlp", "conv_mlp", "dcgan", "wgan", "lsgan"]:
        raise NotImplementedError("The model '{}' is not yet implemented yet, sorry!".format(settings.MODEL))

    ### Setup logging and xtraceback
    try:
        import logging
        try:
            import xtraceback
            xtraceback.compat.install()
            settings.MODULE_HAVE_XTRACEBACK = True
        except ImportError as e:
            print_warning("You do not have the 'xtraceback' module. " +
                          "Please consider installing it, and all other " +
                          "required or optional modules, via the command " +
                          "line \"> pip install -r requirements.txt\"")
    except ImportError as e:
        print_warning("You do not have the 'logging' module. Please consider" +
                      " installing it, and all other required or optional " +
                      "modules, via the command line \"> pip install -r requirements.txt\"")

#    try:
    from run_experiment import run_experiment
        ### All systems are go, let's fire up this ship to infinity and beyond!
    run_experiment()
#    except ImportError as e:
#        handle_error("Failed to import 'run_experiment', the most important module. Debugging time!", e)
