# -*- coding: utf-8 -*-

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
    def __init__(self, hyperparams):
        self.model_name = settings.MODEL
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

    def load_model(self):
        return False

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

        
