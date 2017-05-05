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
import random

import lsgan_architectures
import hyper_params
import settings
from models import GAN_BaseModel
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log

class LSGAN_Model(GAN_BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_lsgan_hyper_params):
        super(LSGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.gen_fn = None
        self.generator_train_fn = None
        self.critic_train_fn = None
        
        # TODO: Turn this into a hyperparameters
        self.optimizer = "rmsprop"
        #self.optimizer = "adam"

        # Constants
        self.gen_filename = "model_generator.npz"
        self.disc_filename = "model_critic.npz"
        self.full_gen_path = os.path.join(settings.MODELS_DIR, self.gen_filename)
        self.full_disc_path = os.path.join(settings.MODELS_DIR, self.disc_filename)

    # ##################### Build the neural network model #######################
    # We create two models: The generator and the critic network.
    # The models are the same as in the Lasagne DCGAN example, except that the
    # discriminator is now a critic with linear output instead of sigmoid output.

    def build_generator(self, input_var=None):
        return lsgan_architectures.build_generator_architecture(input_var, 1)

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

    def train(self, dataset, num_epochs = 1000, epochsize = 50, batchsize = 64, initial_eta = 0.00005,
              architecture = 1):
        import lasagne
        import theano.tensor as T
        from theano import shared, function

        # Load the dataset
        log("Fetching data...")
        X_train, X_test, y_train, y_test, ind_train, ind_test = dataset.return_data()

        # Prepare Theano variables for inputs and targets
        noise_var = T.matrix('noise')
        input_var = T.tensor4('inputs')

        # Create neural network model
        log("Building model and compiling functions...")
        generator = self.build_generator(noise_var)
        critic = self.build_critic(input_var, architecture)

        # Create expression for passing real data through the critic
        real_out = lasagne.layers.get_output(critic)
        # Create expression for passing fake data through the critic
        fake_out = lasagne.layers.get_output(critic, lasagne.layers.get_output(generator))

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
        #generator_updates = lasagne.updates.rmsprop(generator_loss, generator_params, learning_rate=eta)
        #critic_updates = lasagne.updates.rmsprop(critic_loss, critic_params, learning_rate=eta)
        generator_updates = lasagne.updates.adam(generator_loss, generator_params,
                                                 learning_rate=eta, beta1=0.75)
        critic_updates = lasagne.updates.adam(critic_loss, critic_params,
                                              learning_rate=eta, beta1=0.75)

        # Instantiate a symbolic noise generator to use for training
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
        noise = srng.uniform((batchsize, 100))

        # Compile functions performing a training step on a mini-batch (according
        # to the updates dictionary) and returning the corresponding score:
        from theano import function
        generator_train_fn = function([], generator_loss, givens={noise_var: noise}, updates=generator_updates)
        critic_train_fn = function([input_var], critic_loss, givens={noise_var: noise}, updates=critic_updates)

        # Compile another function generating some data
        gen_fn = function([noise_var], lasagne.layers.get_output(generator, deterministic=True))

        # Finally, launch the training loop.
        log("Starting training...")
        # We create an infinite supply of batches (as an iterable generator):
        batches = self.iterate_minibatches(X_train, y_train, batchsize, shuffle=True, forever=True)

        # We iterate over epochs:
        epoch_eta_threshold = num_epochs // 5
        generator_runs = 0
        mean_g_loss = 0
        mean_c_loss = 0
        for epoch in range(num_epochs):
            start_time = time.time()

            if self.check_stop_file():
                print_error("Detected a STOP file. Aborting experiment.")
                break

            # In each epoch, we do `epochsize` generator updates. Usually, the
            # critic is updated 5 times before every generator update. For the
            # first 25 generator updates and every 500 generator updates, the
            # critic is updated 100 times instead, following the authors' code.
            critic_losses = []
            generator_losses = []
            for _ in range(epochsize):
                if mean_c_loss < 0.15:
                    critic_runs = 1
                elif mean_c_loss < mean_g_loss/5.0:
                    critic_runs = 3
                elif mean_c_loss > mean_g_loss:
                    critic_runs = 30
                else:
                    critic_runs = 5
                for _ in range(critic_runs):
                    batch = next(batches)
                    inputs, targets = batch
                    critic_losses.append(critic_train_fn(inputs))
                    generator_runs
                if mean_g_loss > mean_c_loss*5.0:
                    generator_runs = 30
                else:
                    generator_runs = 5
                for _ in range(generator_runs):
                    generator_losses.append(generator_train_fn())

            # Then we print the results for this epoch:
            log("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            mean_g_loss = np.mean(generator_losses)
            mean_c_loss = np.mean(critic_losses)
            log("  generator loss = {}".format(mean_g_loss))
            log("  critic loss    = {}".format(mean_c_loss))

            # And finally, we plot some generated data
            # And finally, we plot some generated data, depending on the settings
            if epoch % settings.EPOCHS_PER_SAMPLES == 0:
                from utils import normalize_data, denormalize_data
                # And finally, we plot some generated data
                # Generate 100 images, which we will output in a 10x10 grid
                samples = np.array(gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100))))
                samples = denormalize_data(samples)
                samples_path = os.path.join(settings.EPOCHS_DIR, 'samples_epoch_{0:0>5}.png'.format(epoch + 1))
                try:
                    import PIL.Image as Image
                    Image.fromarray(samples.reshape(10, 10, 3, 64, 64)
                                    .transpose(0, 3, 1, 4, 2)
                                    .reshape(10*64, 10*64, 3)).save(samples_path)
                except ImportError as e:
                    print_warning("Cannot import module 'PIL.Image', which is necessary for the LSGAN to output its sample images. You should really install it!")

            # After half the epochs, we start decaying the learn rate towards zero
            if epoch >= epoch_eta_threshold:
                progress = float(epoch - epoch_eta_threshold) / float(num_epochs)
                eta.set_value(lasagne.utils.floatX(initial_eta*math.pow(1 - progress, 2)))
            # if epoch >= num_epochs // 2:
            #     progress = float(epoch) / num_epochs
            #     eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

        # Optionally, you could now dump the network weights to a file like this:
        np.savez(os.path.join(settings.MODELS_DIR, 'lsgan_gen.npz'), *lasagne.layers.get_all_param_values(generator))
        np.savez(os.path.join(settings.MODELS_DIR, 'lsgan_crit.npz'), *lasagne.layers.get_all_param_values(critic))
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)

        self.generator = generator
        self.critic = critic
        self.generator_train_fn = generator_train_fn
        self.critic_train_fn = critic_train_fn
        self.gen_fn = gen_fn

        return generator, critic, gen_fn
