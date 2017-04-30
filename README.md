# ift6266-h17-project

Experiments, code, models and notes for the final project of the IFT6266-H2017 "Deep Learning" course by Aaron Courville.

# Example:

./run.py mlp SimpleModel-1 -e 50 -c 10 -k -l 0.0002 -b 128 -v 2
./run.py lsgan MyExpName-2 -e 1000 -u 30 -c 50 -a 1 -s 10 -g 0.0005 -b 64 -v 2 -f
./run.py conv-vgg ConvVgg-3 --feature-matching -e 200 -c 20 -k -l 0.0002 -b 128 -v 2
./run.py lstm LSTM-4 -e 100 -c 10 -l 0.0008 -b 128 -v 2

Running with default values (which may or may not be recommended), you can also try simply:

./run.py mlp test1
./run.py conv-mlp test2
./run conv-vgg test3
./run lstm test4
./run.py lsgan test5
./run.py wgan test6
./run.py dcgan test7

# Installation

## Requirements:

* Theano >= 0.9.0 (ideally, bleeding-edge GPU version 0.9.0.dev5)
* Lasagne >= 0.2.dev1
* Keras >= 2.0.0
* h5py >= 2.6.0
* hdf5 >= 1.8.17

## Optional:

* graphviz >= 0.6
* pydot >= 1.2.3
* pydot-ng >= 1.0.0
* xtraceback >= 0.3.3
* logging >= 0.4.9.6

All required and optional packages can be installed within your virtual environment by typing:

> python -r requirements.txt

# References:

DC-GAN code was tweaked from:
https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56

Convolutional Autoencoder was tweaked from:
https://github.com/mikesj-public/convolutional_autoencoder

W-GAN code was tweaked from:
https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066

# Usage:
usage: run.py [-h] [-v VERBOSE] [-e EPOCHS] [-b BATCH_SIZE] [-l LEARNING_RATE]                                       
              [-g GAN_LEARNING_RATE] [-f] [-c EPOCHS_PER_CHECKPOINT]                                                 
              [-s EPOCHS_PER_SAMPLES] [-k] [-a ARCHITECTURE]                                                         
              [-u UPDATES_PER_EPOCH] [-m FEATURE_MATCHING]                                                           
              model exp_name_prefix                                                                                  
                                                                                                                     
positional arguments:                                                                                                
  model                 Model choice (current options: test, mlp, conv_mlp*,                                         
                        conv_lstm*, vae*, conv_autoencoder*, dcgan, wgan,                                            
                        lsgan (*: Models with * may not be fully implemented                                         
                        yet).)                                                                                       
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
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train (either for a new model or
                        *extra* epochs when resuming an experiment.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Only use this if you want to override the default
                        hyper parameter value for your model. It may be better
                        to create an experiment directory with an
                        'hyperparameters.json' file and tweak the parameters
                        there.
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Only set this if you want to override the default
                        hyper parameter value for your model. It may be better
                        to create an experiment directory with an
                        'hyperparameters.json' file and tweak the parameters
                        there.
  -g GAN_LEARNING_RATE, --gan_learning_rate GAN_LEARNING_RATE
                        Only set this if you want to override the default
                        hyper parameter value for your GAN model. It may be
                        better to create an experiment directory with an
                        'hyperparameters.json' file and tweak the parameters
                        there.
  -f, --force           Force the experiment to run even if a STOP file is
                        present. This will also delete the STOP file.
  -c EPOCHS_PER_CHECKPOINT, --epochs_per_checkpoint EPOCHS_PER_CHECKPOINT
                        Amount of epochs to perform during training between
                        every checkpoint.
  -s EPOCHS_PER_SAMPLES, --epochs_per_samples EPOCHS_PER_SAMPLES
                        Amount of epochs to perform during training between
                        every generation of image samples (typically 100
                        images in a 10x10 grid). If your epochs are rather
                        short, you might want to increase this value, as
                        generating images and saving them to disk can be
                        relatively costly.
  -k, --keep_all_checkpoints
                        By default, only the model saved during the last
                        checkpoint is saved. Pass this flag if you want to
                        keep a models on disk with its associated epoch in the
                        filename at every checkpoint.
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Architecture type, only applies to the LSGAN model
                        (values: 1, 2, 3 or 4).
  -u UPDATES_PER_EPOCH, --updates_per_epoch UPDATES_PER_EPOCH
                        Number of times to update the generator and
                        discriminator/critic per epoch. Applies to GAN models
                        only.
  -m FEATURE_MATCHING, --feature_matching FEATURE_MATCHING
                        By default, feature matching is not used
                        (equivalently, it is set to 0, meaning that the loss
                        function uses the last layer's output). You can set
                        this value to 1 to use the output of the second-to-
                        last layer, or a value of 2 to use the output of the
                        third-to-last layer, and so on. This technique is
                        called 'feature matching' and many provide benefits in
                        some cases. Note that it is not currently implemented
                        in all models and you will receive a message
                        indicating if feature matching is used for your model.

