# -*- coding: utf-8 -*-

import os, sys, errno
import json
import time

import utils
import hyper_params
import settings
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive
from utils import force_symlink, get_json_pretty_print

class BaseModel(object):
    def __init__(self, model_name, hyperparams):
        self.model_name = model_name
        self.model_path = None
        self.current_model_path = None
        self.hyper = hyperparams
        self.epochs_completed = 0
        self.wall_time_start = 0
        self.process_time_start = 0
        self.total_wall_time = 0
        self.total_process_time = 0
        self.resume_from_checkpoint = False
        self.model_compiled = False

    def load_model(self):
        pass
    
    def save_model(self):
        pass

    def save_hyperparams(self):
        settings.touch_dir(settings.BASE_DIR)
        path = os.path.join(settings.BASE_DIR, "hyperparams.json")
        try:
            with open(path, 'w') as fp:
                fp.write(get_json_pretty_print(self.hyper))
        except Exception as e:
            handle_warning("Failed to write hyper parameters file '{0}'.".format(path), e)
            return False
        return True

    def load_hyperparams(self):
        settings.touch_dir(settings.BASE_DIR)
        path = os.path.join(settings.BASE_DIR, "hyperparams.json")
        try:
            with open(path, 'r') as fp:
                self.hyper = json.load(fp)
        except Exception as e:
            handle_warning("Failed to load hyper parameters file '{0}'.".format(path), e)
            print_warning("Using default hyper parameters instead...")
            return False

        print_positive("Loaded hyper parameters from 'hyperparameters.json' file succesfully!")
        return True

    def create_checkpoint(self):
        settings.touch_dir(settings.BASE_DIR)
        chkpoint_path = os.path.join(settings.BASE_DIR, "checkpoint.json")
        # TODO: Add a lot more to checkpoint file, such as:
        # - current training loss
        # - current validation loss
        # - experiment's directory
        checkpoint = {
            "epochs_completed" : self.epochs_completed,
            "model" : settings.MODEL,
            "exp_name" : settings.EXP_NAME,
            "model_path" : os.path.join(settings.MODELS_DIR, "model.hdf5"),
            "current_model_path" : os.path.join(settings.CHECKPOINTS_DIR, "model_epoch{0}.hdf5".format(self.epochs_completed)),
            "wall_time" : self.total_wall_time,
            "process_time" : self.total_process_time
            }
        try:
            with open(chkpoint_path, 'w') as fp:
                fp.write(get_json_pretty_print(checkpoint))
        except Exception as e:
            handle_error("Unable to write checkpoint file '{}'.".format(chkpoint_path), e)
            return False
        return True
        

    def load_checkpoint(self):
        checkpoint = None
        settings.touch_dir(settings.BASE_DIR)
        chkpoint_path = os.path.join(settings.BASE_DIR, "checkpoint.json")
        if os.path.isfile(chkpoint_path):
            print_positive("Found matching checkpoint file: {}".format(chkpoint_path))
            print_info("Verifying that we can parse the checkpoint file correctly...")
            try:
                with open(chkpoint_path, "r") as fp:
                    try:
                        checkpoint = json.load(fp)
                    except ValueError as e:
                        handle_error("Failed to open checkpoint file '{0}'. ".format(chkpoint_path) +
                                     "It does not appear to be a valid JSON file.", e)
                        checkpoint = None
            except IOError as e:
                handle_error("Unable to open checkpoint file '{}' for reading.".format(chkpoint_path), e)
                checkpoint = None

        # Failed to find or open checkpoint file
        if checkpoint == None:
            self.epochs_completed = 0
            self.total_wall_time = 0
            self.total_process_time = 0
            return None

        self.epochs_completed = checkpoint['epochs_completed']
        if checkpoint['model'] != settings.MODEL:
            print_warning("The checkpoint model '{0}' does not match command line argument of '{1}'."
                          .format(checkpoint['model'], settings.MODEL))
            print_info("Discarding checkpoint and starting from scratch.")
            return None
        if checkpoint['exp_name'] != settings.EXP_NAME:
            print_warning("The checkpoint experiment name '{0}' does not match command line argument of '{1}'."
                          .format(checkpoint['exp_name'], settings.EXP_NAME))
            print_info("Discarding checkpoint and starting from scratch.")
            return None

        # Successfully loaded checkpoint file, try to load model
        print_positive("Successfully loaded checkpoint!")
        print_positive("Resuming from the following last valid state:")
        self.model_path = checkpoint['model_path']
        self.current_model_path = checkpoint['current_model_path']
        self.wall_time = checkpoint['wall_time']
        self.process_time = checkpoint['process_time']
        self.resume_from_checkpoint = True

        print_info("Loading latest HDF5 model...")
        if not self.load_model():
            self.build()
            self._compile()
        self.model_compiled = True

        # Successfully loaded checkpoint, so try to load hyper parameters as well
        print_info("Loading hyper parameters JSON file...")
        self.load_hyperparams()
                    
        return checkpoint


    def check_stop_file(self):
        """Return True if a file with name STOP is found in the base directory, return False otherwise"""
        path = os.path.join(settings.BASE_DIR, "STOP")
        return os.path.isfile(path)
                      
    def create_stop_file(self):
        """Adds an file with name STOP within experiment's root directory. This prevents further training unless the file is deleted. This also prevents loading the dataset and performing any pre-processing."""
        settings.touch_dir(settings.BASE_DIR)
        path = os.path.join(settings.BASE_DIR, "STOP")
        open(path, 'a').close()

    def remove_stop_file(self):
        """Deletes the STOP file in order to allow training to be resumed."""
        path = os.path.join(settings.BASE_DIR, "STOP")
        if os.path.isfile(path):
            os.remove(path)

    def build(self):
        raise NotImplemented()

    def train(self):
        raise NotImplemented()

    def predict(self):
        raise NotImplemented()

    def get_stats(self):
        raise NotImplemented()

    def save_stats(self):
        raise NotImplemented()

class KerasModel(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_mlp_hyper_params): 
        super(KerasModel, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.keras_model = None

    def load_model(self):
        """Return True if a valid model was found and correctly loaded. Return False if no model was loaded."""
        from keras.models import load_model

        settings.touch_dir(settings.CHECKPOINTS_DIR)
        settings.touch_dir(settings.BASE_DIR)
        latest_model_path = self.model_path
        if not os.path.isfile(latest_model_path):
            if not os.path.isfile(self.current_model_path):
                latest_model_path = self.current_model_path
                print_warning("Cannot find model's HDF5 file at path '{}'".format(self.current_model_path))
                return False
            
        print_positive("Found latest HDF5 model saved to disk at: {}".format(latest_model_path))
        print_info("Attempting to load model...")
        try:
            open(latest_model_path, "r").close()
        except Exception as e:
            handle_error("Do not have permission to open for reading HDF5 model located at'{}'.".format(latest_model_path), e)
            return False
        
        try:
            self.keras_model = load_model(latest_model_path)
        except Exception as e:
            handle_error("Model '{0}' is not a valid HDF5 Keras model and cannot be loaded.".format(latest_model_path), e)
            return False
        return True

        
    def save_model(self):
        from keras import models

        settings.touch_dir(settings.CHECKPOINTS_DIR)
        settings.touch_dir(settings.MODELS_DIR)

        self.model_path = os.path.join(settings.CHECKPOINTS_DIR, "model_epoch{0}.hdf5".format(self.epochs_completed))
        print_positive("Saving model after epoch #{} to disk: {}.".format(self.epochs_completed, self.model_path))
        self.keras_model.save(self.model_path)

        model_symlink_path = os.path.join(settings.MODELS_DIR, "model.hdf5")
        force_symlink("../checkpoints/model_epoch{0}.hdf5".format(self.epochs_completed), model_symlink_path)

    def _compile(self):
        # Compile model
        if not self.model_compiled:
            from keras import optimizers
            from keras import losses

            print_info("Compiling model...")
            if self.hyper['optimizer'] == "adam":
                optimizer = optimizers.Adam(lr = self.hyper['learning_rate']) # Default lr = 0.001
            else:
                optimizer = self.hyper['optimizer']
                
            self.keras_model.compile(loss = self.hyper['loss_function'],
                                     optimizer = optimizer,
                                     metrics = [self.hyper['loss_function']])
            self.model_compiled = True

    def train(self, Dataset):
        ### Get the datasets
        X_train, X_test, Y_train, Y_test, id_train, id_test = Dataset.return_data()

        #### Print model summary
        print_info("Model summary:")
        print(self.keras_model.summary())

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
                   .format(self.epochs_completed,
                           self.epochs_completed + settings.NUM_EPOCHS,
                           "(i.e. training an extra {0} epochs)".format(settings.NUM_EPOCHS) if self.epochs_completed == 0 else "",
                           settings.EPOCHS_PER_CHECKPOINT))

        print_info("Ready to start training! For convenience, here are the most important parameters we shall be using again.")
        print_positive("TRAINING PARAMETERS:")
        print(" * num_epochs            = {}".format(settings.NUM_EPOCHS))
        print(" * initial_epoch         = {}".format(self.epochs_completed))
        print(" * final_epoch           = {}".format(self.epochs_completed + settings.NUM_EPOCHS - 1))
        print(" * epochs_per_checkpoint = {}".format(settings.EPOCHS_PER_CHECKPOINT))
        print(" * batch_size            = {}".format(self.hyper['batch_size']))
        print(" * optimizer             = {}".format(self.hyper['optimizer']))
        print(" * loss_function         = {}".format(self.hyper['loss_function']))
        print(" * learning_rate         = {}".format(self.hyper['learning_rate']))

        print_positive("Starting to train model...")
        epoch = 0
        next_epoch_checkpoint = settings.EPOCHS_PER_CHECKPOINT
        while epoch < settings.NUM_EPOCHS:
            while epoch < next_epoch_checkpoint:
                epochs_for_this_fit = min( settings.NUM_EPOCHS - epoch, next_epoch_checkpoint - epoch)
                self.keras_model.fit(X_train, Y_train,
                                     validation_data = (X_test, Y_test),
                                     epochs = self.epochs_completed + epochs_for_this_fit,
                                     batch_size = self.hyper['batch_size'],
                                     verbose = settings.VERBOSE,
                                     initial_epoch = self.epochs_completed)
                epoch += epochs_for_this_fit
                self.epochs_completed += epochs_for_this_fit
            # Checkpoint time (save hyper parameters, model and checkpoint file)
            print_positive("CHECKPOINT AT EPOCH {}. Updating 'checkpoint.json' file...".format(self.epochs_completed))
            self.save_model()
            self.create_checkpoint()
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
        settings.touch_dir(settings.MODELS_DIR)
        path_model_score = os.path.join(settings.MODELS_DIR, "model_score.txt")
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
        self.keras_model.add(Dense(units=1024, input_shape=(self.hyper['input_dim'], )))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(units=512))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(Dense(units=self.hyper['output_dim']))

        ### Plot a graph of the model's architecture
        try:
            import pydot
            try:
                import graphviz
                from keras.utils import plot_model
                settings.touch_dir(settings.MODELS_DIR)
                plot_model(self.keras_model, to_file=os.path.join(settings.MODELS_DIR, 'model_plot.png'), show_shapes=True)
            except ImportError as e:
                handle_warning("Module graphviz not found, cannot plot a diagram of the model's architecture.", e)
        except ImportError as e:
            handle_warning("Module pydot not found, cannot plot a diagram of the model's architecture.", e)

        ### Output a summary of the model, including the various layers, activations and total number of weights
        settings.touch_dir(settings.MODELS_DIR)
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
        if K.image_data_format() == 'channels_first':
            input_shape = (3, 64, 64)
        else:
            input_shape = (64, 64, 3)
        self.keras_model = Sequential()
        self.keras_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.keras_model.add(Conv2D(32, (3, 3)))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.keras_model.add(Conv2D(64, (3, 3)))
        self.keras_model.add(Activation('relu'))
        self.keras_model.add(MaxPooling2D(pool_size=(2, 2)))

        self.keras_model.add(Flatten())
        self.keras_model.add(Dense(512))
        self.keras_model.add(Activation('relu'))
    
        self.keras_model.add(Dense(units=model_params.output_dim))

class DCGAN_Model(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_dcgan_hyper_params):
        super(DCGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)

    def train(self, dataset):
        import dcgan_lasagne
        generator, discriminator, train_fn, gen_fn = dcgan_lasagne.train(dataset, num_epochs=settings.NUM_EPOCHS, initial_eta=5e-4)
        return generator, discriminator, train_fn, gen_fn

class WGAN_Model(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_wgan_hyper_params):
        super(WGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)


class LSGAN_Model(BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_lsgan_hyper_params):
        super(LSGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)


