# ift6266-h17-project

Experiments, code, models and notes for the final project of the IFT6266-H2017 "Deep Learning" course by Aaron Courville.

# Requirements:

* theano (bleeding edge GPU version)
* keras
* lasagne

# References:

DC-GAN code was tweaked from:
https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56

Convolutional Autoencoder was tweaked from:
https://github.com/mikesj-public/convolutional_autoencoder

W-GAN code was tweaked from:
https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066

# Usage:
usage: run_experiment.py [-h] [-v VERBOSE] [-e NUM_EPOCHS] [-b BATCH_SIZE]
                         [-l LEARNING_RATE] [-r]
                         model exp_name_prefix

positional arguments:
  model                 Model choice (current options: mlp, conv_mlp,
                        conv_lstm, vae, conv_autoencoder, dcgan, wgan)
  exp_name_prefix       Prefix used at the beginning of the name of the
                        experiment. Your results will be stored in various
                        subfolders and files which start with this prefix. The
                        exact name of the experiment depends on the model used
                        and various hyperparameters.

optional arguments:
  -h, --help            show this help message and exit
  -v VERBOSE, --verbose VERBOSE
                        0 means quiet, 1 means verbose and 2 means limited
                        verbosity.
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of epochs to train
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Size of minibatches
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate of adam optimizer
  -r, --reload_model    Looks for an existing HF5 model saved to disk in the
                        subdirectory 'models' and if such a model with the
                        same parameters and experiment name prefix exist, the
                        training phase will be entirely skipped and rather,
                        the model and its weights will be loaded from disk.
