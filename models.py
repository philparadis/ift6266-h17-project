# -*- coding: utf-8 -*-

import os, sys, errno
import json
import time
import abc, six

import utils
import hyper_params
import settings
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log
from utils import force_symlink, get_json_pretty_print

# @six.add_metaclass(abc.ABCMeta)
# class LearningModel(object):
#     def __init__(self):
#         pass

#     @abc.abstractmethod
#     def load_model(self):
#         pass
    
#     @abc.abstractmethod
#     def save_model(self, keep_all_checkpoints=False):
#         pass    

#     @abc.abstractmethod
#     def build(self):
#         pass

#     @abc.abstractmethod
#     def train(self):
#         pass

#     @abc.abstractmethod
#     def predict(self, test_batch):
#         pass

# class Checkpoint(object):
#     def __init__(self):
#         self.file_path = "checkpoint.json"

#     def exists(self):
#         """Check if 'checkpoint.json' exists within the base directory."""
#         return os.path.isfile(self.file_path)

#     def read(self):
#         pass

#     def write(self):
#         pass

# class HyperParams(object):
#     def __init__(self):

class BaseModel(object):
    def __init__(self, model_name, hyperparams):
        self.model_name = model_name
        self.model_path = None
        self.current_model_path = None
        self.hyper = hyperparams
        self.epochs_completed = 0
        self.wall_time_start = 0
        self.process_time_start = 0
        self.wall_time = 0
        self.process_time = 0
        self.resume_from_checkpoint = False
        self.model_compiled = False

        # Constant variables
        self.path_stop_file = os.path.join(settings.BASE_DIR, "STOP")
        self.checkpoint_filename = "checkpoint.json"
        self.path_checkpoint_file = os.path.join(settings.BASE_DIR, self.checkpoint_filename)
        self.hyperparams_filename = "hyperparams.json"
        self.path_hyperparams_file = os.path.join(settings.BASE_DIR, self.hyperparams_filename)


    def is_there_hyperparams_file(self):
        return os.path.isfile(self.path_hyperparams_file)
    
    def save_hyperparams(self):
        try:
            with open(self.path_hyperparams_file, 'w') as fp:
                fp.write(get_json_pretty_print(self.hyper))
        except Exception as e:
            handle_warning("Failed to write hyper parameters file '{0}'.".format(self.path_hyperparams_file), e)
            return False
        return True

    def load_hyperparams(self):
        try:
            with open(self.path_hyperparams_file, 'r') as fp:
                self.hyper = json.load(fp)
        except Exception as e:
            handle_warning("Failed to load hyper parameters file '{0}'.".format(self.path_hyperparams_file), e)
            print_warning("Using default hyper parameters instead...")
            return None

        return self.hyper

    def is_there_checkpoint_file(self):
        """Check if 'checkpoint.json' exists within the base directory."""
        return os.path.isfile(self.path_checkpoint_file)

    def update_checkpoint(self, keep_all_checkpoints=False):
        """This initiates a checkpoint, including updating 'checkpoint.json', saving whatever appropriate model(s) and other data within the checkpoints directory and so on. If keep_all_checkpoints is set to True, then a copy of the entire model's weight and optimizer state is preserved for each checkpoint, along with the corresponding epoch in the file name. If set to False, then only the latest model is kept on disk, saving a lot of space, but potentially losing a good model due to overtraining."""
        return (self.save_hyperparams()
                and self.write_checkpoint_file()
                and self.save_model(keep_all_checkpoints = keep_all_checkpoints))

    def resume_last_checkpoint(self):
        checkpoint = None
        hyperparams = None
        loaded_model = False
        if self.is_there_checkpoint_file():
            checkpoint = self.read_checkpoint_file()
            
        if self.is_there_hyperparams_file():
            hyperparams = self.load_hyperparams()

        if checkpoint == None or hyperparams == None:
            # Do not attempt to load the model if we don't even have a checkpoint or hyperparams
            self.resume_from_checkpoint = False
            return None, None, False

        loaded_model = self.load_model()

        if checkpoint == None or hyperparams == None or not loaded_model:
            self.resume_from_checkpoint = False
        else:
            self.resume_from_checkpoint = True
            self.model_compiled = True

        return checkpoint, hyperparams, self.resume_from_checkpoint

    def write_checkpoint_file(self):
        """Write the 'checkpoint.json' file."""
        # TODO: Add a lot more to checkpoint file, such as:
        # - current training loss
        # - current validation loss
        # - experiment's directory
        checkpoint = {
            "epochs_completed" : self.epochs_completed,
            "model" : settings.MODEL,
            "exp_name" : settings.EXP_NAME,
            "wall_time" : self.wall_time,
            "process_time" : self.process_time
            }
        try:
            with open(self.path_checkpoint_file, 'w') as fp:
                fp.write(get_json_pretty_print(checkpoint))
        except Exception as e:
            handle_error("Unable to write checkpoint file '{}'.".format(self.path_checkpoint_file), e)
            return False
        return True

    def read_checkpoint_file(self):
        """Read the 'checkpoint.json' file and update the class variables accordingly."""
        checkpoint = None
        if os.path.isfile(self.path_checkpoint_file):
            print_positive("Found checkpoint file: {}".format(self.path_checkpoint_file))
            print_info("Verifying integrity of checkpoint file...")
            try:
                with open(self.path_checkpoint_file, "r") as fp:
                    try:
                        checkpoint = json.load(fp)
                    except ValueError as e:
                        handle_error("Failed to open checkpoint file '{0}'. ".format(self.path_checkpoint_file) +
                                     "It does not appear to be a valid JSON file.", e)
                        checkpoint = None
            except IOError as e:
                handle_error("Unable to open checkpoint file '{}' for reading.".format(self.path_checkpoint_file), e)
        ### Failed to find or open checkpoint file. Set some values to 0 and exit
        if checkpoint != None:
            ### Succesfully loaded check point file, gather the data!
            print_positive("Successfully loaded checkpoint! Reading its data...")
            self.epochs_completed = checkpoint['epochs_completed']
            if checkpoint['model'] != settings.MODEL:
                print_warning("Inconsistency detected: the checkpoint model '{0}' does not match command line argument of '{1}'."
                              .format(checkpoint['model'], settings.MODEL))
                print_info("Discarding checkpoint and starting from scratch.")
                return None
            if checkpoint['exp_name'] != settings.EXP_NAME:
                print_warning("Inconsistency detected: the checkpoint experiment name '{0}' does not match command line argument of '{1}'.".format(checkpoint['exp_name'], settings.EXP_NAME))
                print_info("Discarding checkpoint and starting from scratch.")
                return None

            self.wall_time = checkpoint['wall_time']
            self.process_time = checkpoint['process_time']
        else:
            self.epochs_completed = 0
            self.wall_time = 0
            self.process_time = 0

        return checkpoint

    def check_stop_file(self):
        """Return True if a file with name STOP is found in the base directory, return False otherwise"""
        return os.path.isfile(self.path_stop_file)
                      
    def create_stop_file(self):
        """Adds an file with name STOP within experiment's root directory. This prevents further training unless the file is deleted. This also prevents loading the dataset and performing any pre-processing."""
        open(self.path_stop_file, 'a').close()

    def remove_stop_file(self):
        """Deletes the STOP file in order to allow training to be resumed."""
        if os.path.isfile(self.path_stop_file):
            os.remove(self.path_stop_file)

    @abc.abstractmethod
    def train(self):
        raise NotImplemented()

    def predict(self):
        raise NotImplemented()

    def get_stats(self):
        raise NotImplemented()

    def save_stats(self):
        raise NotImplemented()

    def plot_layers():
        raise NotImplemented()

from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def __str__(self):
        for epoch, loss in enumerate(self.losses):
            print("Epoch {0:>4}/{1:<4} loss = {2:.5f}".format(epoch, settings.NUM_EPOCHS, loss))


class KerasModel(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_mlp_hyper_params): 
        super(KerasModel, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.keras_model = None
        # Feature matching layers
        self.feature_matching_layers = []
        # Constants
        self.model_path = os.path.join(settings.MODELS_DIR, "model.hdf5")
        # Loss history Keras callback
        self.history = LossHistory()
        
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

        model_epoch_filename = "model_epoch_{0:0>4}.hdf5".format(self.epochs_completed)
        most_recent_model_path = os.path.join(settings.CHECKPOINTS_DIR, model_epoch_filename)
        latest_model_path = self.model_path
        if not os.path.isfile(latest_model_path):
            if not os.path.isfile(most_recent_model_path):
                latest_model_path = most_recent_model_path
                print_warning("Unexpected problem: cannot find the model's HDF5 file anymore at path:\n'{}'".format(most_recent_model_path))
                return False
            
        print_positive("Loading last known valid model (this includes the complete architecture, all weights, optimizer's state and so on)!")
        # Check if file is readable first
        try:
            open(latest_model_path, "r").close()
        except Exception as e:
            handle_error("Lacking permission to *open for reading* the HDF5 model located at\n{}."
                         .format(latest_model_path), e)
            return False
        # Load the actual HDF5 model file
        try:
            self.keras_model = load_model(latest_model_path)
        except Exception as e:
            handle_error("Unfortunately, the model did not parse as a valid HDF5 Keras model and cannot be loaded for an unkown reason. A backup of the model will be created, after which training will restart from scratch.".format(latest_model_path), e)
            try:
                copyfile(latest_model_path, "{}.backup".format(latest_model_path))
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
            # if self.hyper['optimizer'] == "adam":
            #     optimizer = optimizers.Adam(lr = self.hyper['learning_rate']) # Default lr = 0.001
            # else:
            #     optimizer = self.hyper['optimizer']
            optimizer = optimizers.Adam(lr = settings.LEARNING_RATE)

            self.keras_model.compile(loss = self.hyper['loss_function'],
                                     optimizer = optimizer,
                                     metrics = [self.hyper['loss_function']])
            self.model_compiled = True

    def train(self, Dataset):
        ### Get the datasets
        X_train, X_test, Y_train, Y_test, id_train, id_test = Dataset.return_data()

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
                           "(i.e. training an extra {0} epochs)".format(settings.NUM_EPOCHS) if self.epochs_completed == 0 else "",
                           settings.EPOCHS_PER_CHECKPOINT))

        print_positive("Starting to train model!...")
        epoch = 0
        next_epoch_checkpoint = settings.EPOCHS_PER_CHECKPOINT
        while epoch < settings.NUM_EPOCHS:
            while epoch < next_epoch_checkpoint and epoch < settings.NUM_EPOCHS:
                epochs_for_this_fit = min( settings.NUM_EPOCHS - epoch, next_epoch_checkpoint - epoch)
                self.keras_model.fit(X_train, Y_train,
                                     validation_data = (X_test, Y_test),
                                     epochs = self.epochs_completed + epochs_for_this_fit,
                                     batch_size = self.hyper['batch_size'],
                                     verbose = settings.VERBOSE,
                                     initial_epoch = self.epochs_completed,
                                     callbacks=[self.history])
                logout(str(history))
                epoch += epochs_for_this_fit
                self.epochs_completed += epochs_for_this_fit
            # Checkpoint time (save hyper parameters, model and checkpoint file)
            print_positive("CHECKPOINT AT EPOCH {}. Updating 'checkpoint.json' file...".format(self.epochs_completed))
            self.update_checkpoint(settings.KEEP_ALL_CHECKPOINTS)
            next_epoch_checkpoint += settings.EPOCHS_PER_CHECKPOINT

        ### Training complete
        print_positive("Training complete!")

        ### Evaluate the model's performance
        print_info("Evaluating model...")
        train_scores = self.keras_model.evaluate(X_train, Y_train, batch_size = self.hyper['batch_size'], verbose = 0)
        test_scores = self.keras_model.evaluate(X_test, Y_test, batch_size = self.hyper['batch_size'], verbose = 0)
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
    def __init__(self, model_name, hyperparams = hyper_params.default_mlp_hyper_params):
        super(MLP_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        
    def build(self):
        from keras.layers.core import Dense, Activation
        from keras.models import Sequential

        self.keras_model = Sequential()
        self.keras_model.add(Dense(units=1024, activation='relu', input_shape=(self.hyper['input_dim'], )))
        self.keras_model.add(Dense(units=512, activation='tanh'))
        self.keras_model.add(Dense(units=self.hyper['output_dim']))

    def plot_architecture(self):
        ### Plot a graph of the model's architecture
        try:
            import pydot
            try:
                import graphviz
                from keras.utils import plot_model
                plot_model(self.keras_model, to_file=os.path.join(settings.MODELS_DIR, 'model_plot.png'), show_shapes=True)
            except ImportError as e:
                handle_warning("Module graphviz not found, cannot plot a diagram of the model's architecture.", e)
        except ImportError as e:
            handle_warning("Module pydot not found, cannot plot a diagram of the model's architecture.", e)

        ### Output a summary of the model, including the various layers, activations and total number of weights
        old_stdout = sys.stdout
        sys.stdout = open(os.path.join(settings.MODELS_DIR, 'model_summary.txt'), 'w')
        self.keras_model.summary()
        sys.stdout.close()
        sys.stdout = old_stdout

class Test_Model(KerasModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_test_hyper_params):
        super(Test_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        
    def build(self):
        from keras.layers.core import Dense, Activation
        from keras.models import Sequential

        self.keras_model = Sequential()
        self.keras_model.add(Dense(units=128, input_shape=(self.hyper['input_dim'], )))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(units=self.hyper['output_dim']))
    

class Conv_MLP(KerasModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_conv_mlp_hyper_params):
        super(Conv_MLP, self).__init__(model_name = model_name, hyperparams = hyperparams)

    def build(self):
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
        from keras.layers.advanced_activations import LeakyReLU

        input_shape = (3, 64, 64)
        self.keras_model = Sequential()a
        self.keras_model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu'))  # num_units: 32*64*64
        self.keras_model.add(Dropout(0.2))
        self.keras_model.add(MaxPooling2D(pool_size=(2, 2))) # out: 32x32

        self.keras_model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))  # num_units: 64*32*32
        self.keras_model.add(Dropout(0.5))
        self.keras_model.add(MaxPooling2D(pool_size=(2, 2))) # out: 16x16

        self.feature_matching_layers.append(Conv2D(256, (5, 5), padding='same', activation='relu'))
        self.keras_model.add(self.feature_matching_layer[-1]) # num_units: 128x16x16
        self.keras_model.add(Dropout(0.5))
        self.keras_model.add(MaxPooling2D(pool_size=(2, 2))) # out: 8x8

        self.keras_model.add(Flatten())
        self.keras_model.add(Dense(units=4096, activation='tanh'))
        self.keras_model.add(Dense(units=self.hyper['output_dim'])) # output_dim = 3*32*32 = 3072

class Conv_Deconv(KerasModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_conv_deconv_hyper_params):
        super(Conv_Deconv, self).__init__(model_name = model_name, hyperparams = hyperparams)

    def build(self):
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Conv2D as Convolution2D
        from keras.layers import Conv2DTranspose as Deconvolution2D

        # Conv
        input_img = Input(shape=(3, 64, 64))
        x = Convolution2D(64, 5, strides=(2, 2), padding='same', activation='relu')(input_img) # out: 32x32
        x = Convolution2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x) # out: 16x16
        x = Convolution2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x) # out: 8x8
        # Deconv
        x = Deconvolution2D(64, 5, padding='same', activation='relu')(x) #out: 8x8
        x = Deconvolution2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x) #out: 16x16
        x = Deconvolution2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x) #out: 32x32
        x = Deconvolution2D(64, 5, padding='same', activation='tanh')(x) #out: 32x32
        self.feature_matching_layers.append(x)
        x = Deconvolution2D(3, 5, padding='same')(x) #out: 32x32
        self.keras_model = Model(input=[input_img], output=x)
        
class GAN_BaseModel(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_gan_basemodel_hyper_params):
        super(GAN_BaseModel, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.generator = None
        self.discriminator = None
        self.train_fn = None
        self.gen_fn = None

        # Feature matching layers
        self.feature_matching_layers = []
        
        # Constants
        self.gen_filename = "model_generator.npz"
        self.disc_filename = "model_discriminator.npz"
        self.full_gen_path = os.path.join(settings.MODELS_DIR, self.gen_filename)
        self.full_disc_path = os.path.join(settings.MODELS_DIR, self.disc_filename)

    def build(self):
        pass

    def load_model(self):
        """Return True if a valid model was found and correctly loaded. Return False if no model was loaded."""
        import numpy as np
        from lasagne.layers import set_all_param_values

        if os.path.isfile(self.full_gen_path) and os.path.isfile(self.full_disc_path):
            print_positive("Found latest '.npz' model's weights files saved to disk at paths:\n{}\n{}".format(self.full_gen_path, self.full_disc_path))
        else:
            print_info("Cannot resume from checkpoint. Could not find '.npz'  weights files, either {} or {}.".format(self.full_gen_path, self.full_disc_path))
            return False
            
        try:
            ### Load the generator model's weights
            print_info("Attempting to load generator model: {}".format(self.full_gen_path))
            with np.load(self.full_gen_path) as fp:
                param_values = [fp['arr_%d' % i] for i in range(len(fp.files))]
            set_all_param_values(self.generator, param_values)

            ### Load the discriminator model's weights
            print_info("Attempting to load generator model: {}".format(self.full_disc_path))
            with np.load(self.full_disc_path) as fp:
                param_values = [fp['arr_%d' % i] for i in range(len(fp.files))]
            set_all_param_values(self.discriminator, param_values)
        except Exception as e:
            handle_error("Failed to read or parse the '.npz' weights files, either {} or {}.".format(self.full_gen_path, self.full_disc_path), e)
            return False
        return True

        
    def save_model(self, keep_all_checkpoints=False):
        """Save model. If keep_all_checkpoints is set to True, then a copy of the entire model's weight and optimizer state is preserved for each checkpoint, along with the corresponding epoch in the file name. If set to False, then only the latest model is kept on disk, saving a lot of space, but potentially losing a good model due to overtraining."""
        import numpy as np
        from lasagne.layers import get_all_param_values
        # Save the gen and disc weights to disk
        if keep_all_checkpoints:
            epoch_gen_path = "model_generator_epoch{0:0>4}.npz".format(self.epochs_completed)
            epoch_disc_path = "model_discriminator_epoch{0:0>4}.npz".format(self.epochs_completed)
            chkpoint_gen_path = os.path.join(settings.CHECKPOINTS_DIR, epoch_gen_path)
            chkpoint_disc_path = os.path.join(settings.CHECKPOINTS_DIR, epoch_disc_path)
            np.savez(chkpoint_gen_path, *get_all_param_values(self.generator))
            np.savez(chkpoint_disc_path, *get_all_param_values(self.discriminator))
            force_symlink("../checkpoints/{}".format(epoch_gen_path), self.full_gen_path)
            force_symlink("../checkpoints/{}".format(epoch_disc_path), self.full_disc_path)
        else:
            np.savez(self.full_gen_path, *get_all_param_values(self.generator))
            np.savez(self.full_disc_path, *get_all_param_values(self.discriminator))

