#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Comment by Philippe Paradis. I am NOT the other of the code in this file.
### Source : 

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Least Squares Generative Adversarial Networks
(LSGANs, see https://arxiv.org/abs/1611.04076 for the paper).

It is based on a WGAN example:
https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066
This, in turn, is based on a DCGAN example:
https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
This, in turn, is based on the MNIST example in Lasagne:
https://lasagne.readthedocs.io/en/latest/user/tutorial.html

Jan Schlüter, 2017-03-07
"""

from __future__ import print_function

import sys, os, time
import numpy as np
import math

import lsgan_architectures
import hyper_params
import settings
from models import GAN_BaseModel
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive

class LSGAN_Model(GAN_BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_lsgan_hyper_params):
        super(LSGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self._W_std = 0.02

    def initialize(self):
        self.gen_path = "generator.npy"
        self.disc_path = "critic.npy"
        self.train_fn = None
        self.gen_fn = None

        # TODO: Turn this into a hyperparameters
        self.optimizer = "rmsprop"
        #self.optimizer = "adam"
        

    # ##################### Build the neural network model #######################
    # We create two models: The generator and the critic network.
    # The models are the same as in the Lasagne DCGAN example, except that the
    # discriminator is now a critic with linear output instead of sigmoid output.

    def build_generator(self, input_var=None, architecture=1):
        return lsgan_architectures.build_generator_architecture(input_var, architecture)

    def build_critic(self, input_var=None, architecture=1):
        return lsgan_architectures.build_critic_architecture(input_var, architecture)
   

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False,
                            forever=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)
                yield inputs[excerpt], targets[excerpt]
            if not forever:
                break


    # ############################## Main program ################################
    # Everything else will be handled in our main program now. We could pull out
    # more functions to better separate the code, but it wouldn't make it any
    # easier to read.

    def _pre_train(self):
        pass

    def _gan_train(self):
        pass

    def _post_train(self):
        pass

    def train(self, dataset, num_epochs = 1000, epochsize = 50, batchsize = 64, initial_eta = 0.0003, architecture = 2):
        """You can choose architecture = 1, 2, 3, 4 or 5."""
        import lasagne
        import theano.tensor as T
        from theano import shared, function
        # Load the dataset
        print("Loading data...")

        X_train, X_val, y_train, y_val, ind_train, ind_test = dataset.return_data()
        settings.TRAINING_BATCH_SIZE = X_train.shape[0]

        # Prepare Theano variables for inputs and targets
        noise_var = T.matrix('noise')
        input_var = T.tensor4('inputs')

        # Create neural network model
        print("Building model and compiling functions...")
        generator = self.build_generator(noise_var, architecture = architecture)
        critic = self.build_critic(input_var, architecture = architecture))

        # Create expression for passing real data through the critic
        real_out = lasagne.layers.get_output(critic)
        # Create expression for passing fake data through the critic
        fake_out = lasagne.layers.get_output(critic,
                                             lasagne.layers.get_output(generator))

        # Create loss expressions to be minimized
        # a, b, c = -1, 1, 0  # Equation (8) in the paper
        a, b, c = 0, 1, 1  # Equation (9) in the paper
        generator_loss = lasagne.objectives.squared_error(fake_out, c).mean()
        critic_loss = (lasagne.objectives.squared_error(real_out, b).mean() +
                       lasagne.objectives.squared_error(fake_out, a).mean())

        # Create update expressions for training
        from theano import shared
        generator_params = lasagne.layers.get_all_params(generator, trainable=True)
        critic_params = lasagne.layers.get_all_params(critic, trainable=True)
        eta = shared(lasagne.utils.floatX(initial_eta))
        if self.optimizer == "rmsprop":
            generator_updates = lasagne.updates.rmsprop(
                generator_loss, generator_params, learning_rate=eta)
            critic_updates = lasagne.updates.rmsprop(
                critic_loss, critic_params, learning_rate=eta)
        else: #adam
            generator_updates = lasagne.updates.adam(
                generator_loss, generator_params, learning_rate=eta, beta1=0.5)
            critic_updates = lasagne.updates.adam(
                critic_loss, critic_params, learning_rate=eta, beta1=0.5)

        # Instantiate a symbolic noise generator to use for training
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
        noise = srng.uniform((batchsize, 100))

        # Compile functions performing a training step on a mini-batch (according
        # to the updates dictionary) and returning the corresponding score:
        from theano import function
        generator_train_fn = function([], generator_loss,
                                      givens={noise_var: noise},
                                      updates=generator_updates)
        critic_train_fn = function([input_var], critic_loss,
                                   givens={noise_var: noise},
                                   updates=critic_updates)

        # Compile another function generating some data
        gen_fn = function([noise_var], lasagne.layers.get_output(generator, deterministic=True))
        ###########################################################################
        ###########################################################################
        ###########################################################################

        # If we have time, add the function to resume training from checkpoints,
        # create regular checkpoints and so on.

        ###########################################################################
        ###########################################################################
        ###########################################################################

        # Create experiment's results directories
        settings.touch_dir(settings.BASE_DIR)
        settings.touch_dir(settings.EPOCHS_DIR)
        settings.touch_dir(settings.CHECKPOINTS_DIR)
        settings.touch_dir(settings.MODELS_DIR)

        # Finally, launch the training loop.
        print("Starting training...")
        # We create an infinite supply of batches (as an iterable generator):
        batches = self.iterate_minibatches(X_train, y_train, batchsize, shuffle=True,
                                      forever=True)
        # We iterate over epochs:
        generator_updates = 0
        next_epoch_checkpoint = settings.EPOCHS_PER_CHECKPOINT        
        for epoch in range(num_epochs):
            start_time = time.time()

            # In each epoch, we do `epochsize` generator and critic updates.
            critic_losses = []
            generator_losses = []
            for _ in range(epochsize):
                inputs, targets = next(batches)
                critic_losses.append(critic_train_fn(inputs))
                generator_losses.append(generator_train_fn())

            # Then we print the results for this epoch:
            time_delta = time.time() - start_time
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time_delta))
            print("  generator loss: {}".format(np.mean(generator_losses)))
            print("  critic loss:    {}".format(np.mean(critic_losses)))
            self.wall_time += time_delta
            # TODO: Append performance to a file

            # And finally, we plot some generated data
            from utils import normalize_data, denormalize_data
            from utils import print_warning
            # And finally, we plot some generated data
            # Generate 100 images, which we will output in a 10x10 grid
            samples = np.array(gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100))))
            samples = denormalize_data(samples)
            samples_path = os.path.join(settings.EPOCHS_DIR, 'samples_epoch_{0:0>5}.png'.format(epoch + 1))
            try:
                import PIL.Image as Image
            except ImportError as e:
                print_warning("Cannot import module 'PIL.Image', which is necessary for the LSGAN to output its sample images. You should really install it!")
            else:
                Image.fromarray(samples.reshape(10, 10, 3, 64, 64)
                                .transpose(0, 3, 1, 4, 2)
                                .reshape(10*64, 10*64, 3)).save(samples_path)
                # for ind in range(10):
                #     # Generate a single image
                #     sample = np.array(gen_fn(lasagne.utils.floatX(np.random.rand(1, 100))))
                #     sample = denormalize_data(sample)
                #     sample_path = os.path.join(settings.EPOCHS_DIR,
                #                                'one_sample_epoch_{0:0>5}_num{1}.png'.format(epoch + 1, ind))
                #     Image.fromarray(sample.reshape(3, 64, 64)
                #                     .transpose(1, 2, 0)
                #                     .reshape(64, 64, 3)).save(sample_path)

            if epoch >= next_epoch_checkpoint:
                ### Checkpoint time!!! (save model and checkpoint file)
                print_positive("CHECKPOINT AT EPOCH {}. Updating 'checkpoint.json' file and saving model...".format(epoch + 1))
                self.generator = generator
                self.discriminator = critic
                self.epochs_completed = epoch

                # Create checkpoint
                self.create_checkpoint()
                ## Save model to disk
                self.save_model(latest_only = (not settings.KEEP_ALL_CHECKPOINTS))

                ### Save the model's performance to disk
                path_model_score = os.path.join(settings.CHECKPOINTS_DIR, "score_epoch_{0:0>5}.txt".format(epoch + 1))
                print_info("Saving performance to file '{}'".format(path_model_score))
                with open(path_model_score, "w") as fd:
                    fd.write("Performance statistics\n")
                    fd.write("----------------------\n")
                    fd.write("Model           = {}\n".format(settings.MODEL))
                    fd.write("Experiment name = {}\n".format(settings.EXP_NAME))
                    fd.write("Total epochs    = {0}\n".format(self.epochs_completed))
                    fd.write("Total time      = {0:.2f} seconds\n".format(self.wall_time))
                    fd.write("generator loss  = {}\n".format(np.mean(generator_losses)))
                    fd.write("critic loss     = {}\n".format(np.mean(critic_losses)))
                # Set next checkpoint epoch
                next_epoch_checkpoint = epoch + settings.EPOCHS_PER_CHECKPOINT

            
            # After half the epochs, we start decaying the learn rate towards zero
            if epoch >= num_epochs // 2:
                progress = float(epoch) / float(num_epochs)
                eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))
                lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))

        ### Save model to class variables
        self.generator = generator
        self.discriminator = critic
        self.critic_train_fn = critic_train_fn
        self.generator_train_fn = critic_generator_fn
        self.gen_fn = gen_fn

        ### Save model to disk
        self.save_model()

        return generator, critic, train_fn, gen_fn

