import os, sys, errno
import json
import time
import abc, six
import glob

import utils
import hyper_params
import settings
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log, logout
from utils import force_symlink, get_json_pretty_print

from models import BaseModel

class KerasModel(BaseModel):
    def __init__(self, hyperparams = hyper_params.default_mlp_hyper_params): 
        super(KerasModel, self).__init__(hyperparams = hyperparams)
        self.keras_model = None
        # Feature matching layers
        self.feature_matching_layers = []
        # Constants
        self.model_path = os.path.join(settings.MODELS_DIR, "model.hdf5")
        
    def get_intermediate_activations(model, k, X_batch):
        """Get the (intermediate) activations at the k-th layer in 'model' from input X_batch"""
        get_activations = theano.function([model.layers[0].input],
                                          model.layers[k].get_output(train=False),
                                          allow_input_downcast=True)
        activations = get_activations(X_batch) # same result as above
        return activations
        
    def load_model(self):
        """Return True if a valid model was found and correctly loaded. Return False if no model was loaded."""
        from shutil import copyfile
        from keras.models import load_model

        chosen_model_path = None
        best_model_path = None
        best_model_glob = glob.glob(settings.MODELS_DIR + "/best_model_*.hdf5")
        if len(best_model_glob) > 0:
            best_model_path = best_model_glob[0]
            chosen_model_path = best_model_path
            print_positive("Loading *best* model with the lowest validation score: {}"
                           .format(best_model_path))
        if best_model_path == None:
            model_epoch_filename = "model_epoch_{0:0>4}.hdf5".format(self.epochs_completed)
            most_recent_model_path = os.path.join(settings.CHECKPOINTS_DIR, model_epoch_filename)
            chosen_model_path = self.model_path
            if not os.path.isfile(chosen_model_path):
                if not os.path.isfile(most_recent_model_path):
                    print_warning("Unexpected problem: cannot find the model's HDF5 file anymore at path:\n'{}'".format(most_recent_model_path))
                    return False
                else:
                    chosen_model_path = most_recent_model_path

            print_positive("Loading last known valid model (this includes the complete architecture, all weights, optimizer's state and so on)!")

        # Check if file is readable first
        try:
            open(chosen_model_path, "r").close()
        except Exception as e:
            handle_error("Lacking permission to *open for reading* the HDF5 model located at\n{}."
                         .format(chosen_model_path), e)
            return False
        
        # Load the actual HDF5 model file
        try:
            self.keras_model = load_model(chosen_model_path)
        except Exception as e:
            handle_error("Unfortunately, the model did not parse as a valid HDF5 Keras model and cannot be loaded for an unkown reason. A backup of the model will be created, after which training will restart from scratch.".format(chosen_model_path), e)
            try:
                copyfile(chosen_model_path, "{}.backup".format(chosen_model_path))
            except Exception as e:
                handle_error("Looks like you're having a bad day. The copy operation failed for an unknown reason. We will exit before causing some serious damage ;). Better luck next time. Please verify your directory permissions and your default umask!.", e)
                sys.exit(-3)
            return False
        return True
        
    def save_model(self, keep_all_checkpoints=False):
        """Save model. If keep_all_checkpoints is set to True, then a copy of the entire model's weight and optimizer state is preserved for each checkpoint, along with the corresponding epoch in the file name. If set to False, then only the latest model is kept on disk, saving a lot of space, but potentially losing a good model due to overtraining."""
        from keras import models
        model_epoch_filename = "model_epoch_{0:0>4}.hdf5".format(self.epochs_completed)

        if keep_all_checkpoints:
            chkpoint_model_path = os.path.join(settings.CHECKPOINTS_DIR, model_epoch_filename)
            print_positive("Saving model after epoch #{} to disk:\n{}.".format(self.epochs_completed, chkpoint_model_path))
            self.keras_model.save(chkpoint_model_path)
            force_symlink("../checkpoints/{}".format(model_epoch_filename), self.model_path)
        else:
            print_positive("Overwriting latest model after epoch #{} to disk:\n{}.".format(self.epochs_completed, self.model_path))
            self.keras_model.save(self.model_path)
        return True

    def _compile(self):
        # Compile model
        if not self.model_compiled:
            from keras import optimizers
            from keras import losses

            print_info("Compiling model...")

            optimizer = optimizers.Adam(lr = settings.LEARNING_RATE)

            self.keras_model.compile(loss = settings.LOSS_FUNCTION,
                                     optimizer = optimizer,
                                     metrics = [settings.LOSS_FUNCTION])
            self.model_compiled = True

    def increment_epochs_completed(self, epoch, logs):
        self.epochs_completed += 1
        log("")
        log("Epoch {0:>4}/{1:<4}: loss = {2:.4f}, val_loss = {3:.4f}".
            format(epoch, settings.NUM_EPOCHS, logs['loss'], logs['val_loss']))

    def train(self, Dataset):
        from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
        
        ### Get the datasets
        X_train, X_test, Y_train, Y_test, id_train, id_val = Dataset.return_train_data()

        #### Print model summary
        #print_info("Model summary:")
        #self.keras_model.summary()

        #### Compile the model (if necessary)
        self._compile()
        
        #### Fit the model
        
        # We fit the model iteratively, typically in more than one pass, in order to
        # create frequent checkpoints. For example, if EPOCHS_PER_CHECKPOINT is set to 5
        # and NUM_EPOCHS is set to 33, then the loop will iterate 6 times, performing 5
        # epochs of training via the 'fit' method and on the 7th iteration, it will
        # perform 3 epochs only in order to reach NUM_EPOCHS. Finally, a checkpoint is
        # always created once training is complete, even if the next EPOCHS_PER_CHECKPOINT
        # multiple was not reached yet.

        ### Print the major params again, for convenience
        print_info("Starting training from epoch {0} to epoch {1} {2}, creating checkpoints every {3} epochs."
                   .format(self.epochs_completed + 1,
                           self.epochs_completed + settings.NUM_EPOCHS,
                           "(i.e. training an extra {0} epochs)".format(settings.NUM_EPOCHS),
                           settings.EPOCHS_PER_CHECKPOINT))

        # Define training callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        best_model_path = os.path.join(settings.MODELS_DIR,
                                       "best_model_epoch.{epoch:03d}_loss.{val_loss:.4f}.hdf5")
        checkpointer = ModelCheckpoint(filepath=best_model_path,
                                       verbose=1, save_best_only=True)
        epoch_complete = LambdaCallback(on_epoch_end = self.increment_epochs_completed)

        # Ready to train!
        print_positive("Starting to train model!...")
        epoch = 0
        verbose = settings.VERBOSE
        if verbose == 2:
            verbose = 0 # If verbose == 2, the 'epoch_complete' callback will already be printing the same
        self.keras_model.fit(X_train, Y_train,
                             validation_split=0.1,
                             epochs = settings.NUM_EPOCHS,
                             batch_size = settings.BATCH_SIZE,
                             verbose = verbose,
                             initial_epoch = self.epochs_completed,
                             callbacks=[early_stopping, checkpointer, epoch_complete])
        
        ### Training complete
        print_positive("Training complete!")

        # Checkpoint time (save hyper parameters, model and checkpoint file)
        print_positive("CHECKPOINT AT EPOCH {}. Updating 'checkpoint.json' file...".format(self.epochs_completed))
        self.update_checkpoint(False)

        ### Evaluate the model's performance
        print_info("Evaluating model...")
        train_scores = self.keras_model.evaluate(X_train, Y_train, batch_size = settings.BATCH_SIZE, verbose = 0)
        test_scores = self.keras_model.evaluate(X_test, Y_test, batch_size = settings.BATCH_SIZE, verbose = 0)
        metric = self.keras_model.metrics_names[1]
        print_positive("Training score {0: >6}: {1:.5f}".format(metric, train_scores[1]))
        print_positive("Testing score  {0: >6}: {1:.5f}".format(metric, train_scores[1]))

        ### Save the model's performance to disk
        path_model_score = os.path.join(settings.PERF_DIR, "score.txt")
        print_info("Saving performance to file '{}'".format(path_model_score))
        with open(path_model_score, "w") as fd:
            fd.write("Performance statistics\n")
            fd.write("----------------------\n")
            fd.write("Model = {}\n".format(settings.MODEL))
            fd.write("Cumulative number of training epochs = {0}\n".format(self.epochs_completed))
            fd.write("Training score (metric: {0: >6}) = {1:.5f}\n".format(metric, train_scores[1]))
            fd.write("Testing score  (metric: {0: >6}) = {1:.5f}\n".format(metric, train_scores[1]))
        
    def predict(self, test_data, batch_size):
        return self.keras_model.predict(test_data, batch_size = batch_size)

class MLP_Model(KerasModel):
    def __init__(self, hyperparams = hyper_params.default_mlp_hyper_params):
        super(MLP_Model, self).__init__(hyperparams = hyperparams)
        
    def build(self):
        from keras.layers.core import Dense, Activation
        from keras.models import Sequential

        self.keras_model = Sequential()
        self.keras_model.add(Dense(512, input_shape=(self.hyper['input_dim'], )))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(256))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(self.hyper['output_dim']))
        self.keras_model.add(Activation('sigmoid'))

    def plot_architecture(self):
        pass
        ### Plot a graph of the model's architecture
        # try:
        #     import pydot
        #     try:
        #         import graphviz
        #         from keras.utils import plot_model
        #         plot_model(self.keras_model, to_file=os.path.join(settings.MODELS_DIR, 'model_plot.png'), show_shapes=True)
        #     except ImportError as e:
        #         handle_warning("Module graphviz not found, cannot plot a diagram of the model's architecture.", e)
        # except ImportError as e:
        #     handle_warning("Module pydot not found, cannot plot a diagram of the model's architecture.", e)

        ### Output a summary of the model, including the various layers, activations and total number of weights
        # old_stdout = sys.stdout
        # sys.stdout = open(os.path.join(settings.MODELS_DIR, 'model_summary.txt'), 'w')
        # self.keras_model.summary()
        # sys.stdout.close()
        # sys.stdout = old_stdout

class Test_Model(KerasModel):
    def __init__(self, hyperparams = hyper_params.default_test_hyper_params):
        super(Test_Model, self).__init__(hyperparams = hyperparams)
        
    def build(self):
        from keras.layers.core import Dense, Activation
        from keras.models import Sequential

        self.keras_model = Sequential()
        self.keras_model.add(Dense(units=128, input_shape=(self.hyper['input_dim'], )))
        self.keras_model.add(Dense(128, activation='relu'))
        self.keras_model.add(Dense(self.hyper['output_dim']))
    

class Conv_MLP(KerasModel):
    def __init__(self, hyperparams = hyper_params.default_conv_mlp_hyper_params):
        super(Conv_MLP, self).__init__(hyperparams = hyperparams)

    def build(self):
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout

        input_shape = (3, 64, 64)
        self.keras_model = Sequential()
        self.keras_model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                    input_shape=input_shape, activation='relu'))  # num_units: 32*64*64

        self.keras_model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu'))  # num_units: 64*32*32

        #self.feature_matching_layers.append(Conv2D(256, (5, 5), padding='same', activation='relu'))
        #self.keras_model.add(self.feature_matching_layers[-1]) # num_units: 128x16x16
        self.keras_model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')) # num_units: 128x16x16

        self.keras_model.add(Flatten())
        self.keras_model.add(Dense(units=1024, activation='relu'))
        self.keras_model.add(Dense(self.hyper['output_dim'], activation='sigmoid')) # output_dim = 3*32*32 = 3072
        
        # input_shape = ((None, 3, 64, 64), )
        # self.keras_model = Sequential()
        # self.keras_model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu'))  # num_units: 32*64*64
        # self.keras_model.add(Dropout(0.2))
        # self.keras_model.add(MaxPooling2D(pool_size=(2, 2))) # out: 32x32

        # self.keras_model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))  # num_units: 64*32*32
        # self.keras_model.add(Dropout(0.5))
        # self.keras_model.add(MaxPooling2D(pool_size=(2, 2))) # out: 16x16

        # #self.feature_matching_layers.append(Conv2D(256, (5, 5), padding='same', activation='relu'))
        # #self.keras_model.add(self.feature_matching_layers[-1]) # num_units: 128x16x16
        # self.keras_model.add(Conv2D(256, (5, 5), padding='same', activation='relu')) # num_units: 128x16x16
        # self.keras_model.add(Dropout(0.5))
        # self.keras_model.add(MaxPooling2D(pool_size=(2, 2))) # out: 8x8

        # self.keras_model.add(Flatten())
        # self.keras_model.add(Dense(units=4096, activation='relu'))
        # self.keras_model.add(Dense(self.hyper['output_dim'])) # output_dim = 3*32*32 = 3072

class Conv_Deconv(KerasModel):
    def __init__(self, hyperparams = hyper_params.default_conv_deconv_hyper_params):
        super(Conv_Deconv, self).__init__(hyperparams = hyperparams)

    def build(self):
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Conv2D as Convolution2D
        from keras.layers import Conv2DTranspose as Deconvolution2D

        # Conv
        input_img = Input(shape=(3, 64, 64))
        x = Convolution2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(input_img) # out: 32x32
        x = Convolution2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x) # out: 16x16
        x = Convolution2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x) # out: 8x8
        # Deconv
        x = Deconvolution2D(64, (5, 5), padding='same', activation='relu', data_format="channels_first")(x) #out: 8x8
        x = Deconvolution2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu', data_format="channels_first")(x) #out: 16x16
        x = Deconvolution2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu', data_format="channels_first")(x) #out: 32x32
        x = Deconvolution2D(64, (5, 5), padding='same', activation='relu', data_format="channels_first")(x) #out: 32x32
        self.feature_matching_layers.append(x)
        x = Deconvolution2D(3, (5, 5), padding='same', activation='sigmoid', data_format="channels_first")(x) #out: 32x32
        self.keras_model = Model(inputs=input_img, outputs=x)

        print(self.keras_model.summary())

def Keras_VGG_16(KerasModel):
    def build(self, weights_path=None):
        from matplotlib import pyplot as plt
        import theano
        import cv2
        import numpy as np
        import scipy as sp
        from keras.models import Sequential
        from keras.layers.core import Flatten, Dense, Dropout
        from keras.layers.convolutional import Convolution2D, MaxPooling2D
        from keras.layers.convolutional import ZeroPadding2D
        from keras.optimizers import SGD
        from sklearn.manifold import TSNE
        from sklearn import manifold
        from sklearn import cluster
        from sklearn.preprocessing import StandardScaler

        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), stride=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), stride=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), stride=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), stride=(2,2)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), stride=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        if self.weights_path:
            model.load_weights(self.weights_path)
        return model



