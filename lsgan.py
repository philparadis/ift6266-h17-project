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

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import models
from gan_lasagne import GAN_BaseModel
from utils import normalize_data, denormalize_data

class LSGAN_Model(GAN_BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_lsgan_hyper_params):
        super(LSGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)

    def initialize()
        self.gen_path = "generator.npy"
        self.disc_path = "critic.npy"

    # ##################### Build the neural network model #######################
    # We create two models: The generator and the critic network.
    # The models are the same as in the Lasagne DCGAN example, except that the
    # discriminator is now a critic with linear output instead of sigmoid output.
    def build_generator(self, input_var=None):
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, DropoutLayer
        try:
            from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        except ImportError:
            raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                              "version: http://lasagne.readthedocs.io/en/latest/"
                              "user/installation.html#bleeding-edge-version")
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        from lasagne.nonlinearities import sigmoid
        from lasagne.nonlinearities import LeakyRectify
        lrelu = LeakyRectify(0.2)
        # input: 100dim
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 1024))
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 256*4*4))
        layer = ReshapeLayer(layer, ([0], 256, 4, 4))
        ### four fractional-stride convolutions
        # Note: Apply dropouts in G. See tip #17 from "ganhacks"
        layer = batch_norm(Deconv2DLayer(layer, 192, 7, stride=2, crop='same',
                                         output_size=8, nonlinearity=lrelu))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 128, 7, stride=2, crop='same',
                                         output_size=16, nonlinearity=lrelu))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=2, crop='same',
                                         output_size=32, nonlinearity=lrelu))
        layer = DropoutLayer(layer, p=0.5)
        layer =            Deconv2DLayer(layer, 3, 5, stride=2, crop='same',
                                         output_size=64, nonlinearity=sigmoid)
        print ("Generator output:", layer.output_shape)
        return layer

    def build_generator_architecture2(self, input_var=None):
        from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, DropoutLayer
        try:
            from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        except ImportError:
            raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                              "version: http://lasagne.readthedocs.io/en/latest/"
                              "user/installation.html#bleeding-edge-version")
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        from lasagne.nonlinearities import sigmoid
        from lasagne.nonlinearities import LeakyRectify
        import theano.tensor as T

        # Define some variables
        # MOCKING: Right now we are "mocking" the hyper parameters, but layer one we will use the user-provided values
        # TODO: Turn these functions into methods of a class which derived from a BaseModel class
        input_noise = True
        activation = LeakyRectify(0.2)
        lrelu = LeakyRectify(0.2)
        dropout = True
        # TODO: Change this so that accessing a key which doesn't exist doesn't trigger an
        # unhandled exception, crashing our program.
        # if self.hyper['input_noise']:
        #     input_noise = True
        # if self.hyper['activation'] == "relu":
        #     activation = lasagne.nonlinearities.rectify
        # elif self.hyper['activation'] == "lrelu":
        #     activation = LeakyRectify(0.2)
        # if self.hyper['dropout'] == True:
        #     dropout = True

        # Build the network's layers
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # Injecting some noise after input layer
        if input_noise:
            layer = GaussianNoiseLayer(layer, sigma=0.2)
        # fully-connected layer
        # TODO: Do we need this layer???
        #layer = batch_norm(DenseLayer(layer, 1024))
        # project and reshape
        layer = batch_norm(DenseLayer(layer, num_units = 512*4*4, W=Normal(0.05), nonlinear=lrelu, g=None))
        layer = ReshapeLayer(layer, ([0], 512, 4, 4))
        # Deconvs and batch norms
        layer = batch_norm(Deconv2DLayer(layer, (None, 256, 8, 8), (5, 5), W=Normal(0.05), nonlinearity=lrelu, g=None))
        if dropout:
            layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, (None, 128, 16, 16), (5, 5), W=Normal(0.05), nonlinearity=lrelu, g=None))
        if dropout:
            layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, (None, 64, 32, 32), (5, 5), W=Normal(0.05), nonlinearity=lrelu, g=None))
        if dropout:
            layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, (None, 3, 64, 64), (5, 5), W=Normal(0.05), nonlinearity=T.tanh, train_g=True, init_stdv=0.1))
        gen_dat = lasagne.layers.get_output(layer)
        return layer

    def build_critic(self, input_var=None):
        from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                    DenseLayer, GaussianNoiseLayer)
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        from lasagne.nonlinearities import LeakyRectify

        # MOCKING: Right now we are "mocking" the hyper parameters, but layer one we will use the user-provided values
        # TODO: Turn these functions into methods of a class which derived from a BaseModel class
        input_noise = True
        activation = LeakyRectify(0.2)
        dropout = True
        # TODO: Change this so that accessing a key which doesn't exist doesn't trigger an
        # unhandled exception, crashing our program.
        # if self.hyper['input_noise']:
        #     input_noise = True
        # if self.hyper['activation'] == "relu":
        #     activation = lasagne.nonlinearities.rectify
        # elif self.hyper['activation'] == "lrelu":
        #     activation = LeakyRectify(0.2)
        # if self.hyper['dropout'] == True:
        #     dropout = True

        # input: (None, 3, 64, 64)
        layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
        # Injecting some noise after input layer
        if input_noise:
            layer = GaussianNoiseLayer(layer, sigma=0.2)
        # four convolutions
        layer = batch_norm(Conv2DLayer(layer, 96, 5, stride=2, pad='same',
                                       nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                       nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 192, 7, stride=2, pad='same',
                                       nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 256, 7, stride=2, pad='same',
                                       nonlinearity=activation))
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 512, nonlinearity=activation))
        # output layer (linear)
        layer = DenseLayer(layer, 1, nonlinearity=None)
        print ("critic output:", layer.output_shape)
        return layer


    def build_critic_architecture2(self):
        from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                    DenseLayer, GaussianNoiseLayer)
        try:
            from lasagne.layers.dnn import batch_norm_dnn as batch_norm
        except ImportError:
            from lasagne.layers import batch_norm
        from lasagne.nonlinearities import LeakyRectify
        from lasagne.init import Normal

        # MOCKING: Right now we are "mocking" the hyper parameters, but layer one we will use the user-provided values
        # TODO: Turn these functions into methods of a class which derived from a BaseModel class
        input_noise = True
        activation = LeakyRectify(0.2)
        dropout = True
        # TODO: Change this so that accessing a key which doesn't exist doesn't trigger an
        # unhandled exception, crashing our program.
        # if self.hyper['input_noise']:
        #     input_noise = True
        # if self.hyper['activation'] == "relu":
        #     activation = lasagne.nonlinearities.rectify
        # elif self.hyper['activation'] == "lrelu":
        #     activation = LeakyRectify(0.2)
        # if self.hyper['dropout'] == True:
        #     dropout = True

        # input: (None, 3, 64, 64)
        layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
        # new layer: What is it for???
        # ganhacks says:
        # * Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
        #    - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        #    - https://openreview.net/forum?id=Hk4_qw5xe
        # * Adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
        #    - (Zhao et. al. EBGAN)
        #    - "Improved GANs" by OpenAI. Their code also has it, but is commented out
        if input_noise:
            layer = GaussianNoiseLayer(layer, sigma=0.2)

        # 3x convolutions with 96 filters each and 3x3 receptive field
        # 1st, 2nd conv preserve dimension
        # 3rd conv has stride 2, so it downsamples dimension by a factor of 2
        layer = batch_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, W=Normal(0.05), nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, W=Normal(0.05), nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, stride=2, W=Normal(0.05), nonlinearity=activation))
        # followed by Dropout p=0.5
        if dropout:
            layer = DropoutLayer(layer, p=0.5)
        # 3x convolutions with 192 filters each and 3x3 receptive field
        # 1st, 2nd conv preserve dimension
        # 3rd conv has stride 2, so it downsamples dimension by a factor of 2
        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, W=Normal(0.05), nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, W=Normal(0.05), nonlinearity=activation))
        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, stride=2, W=Normal(0.05), nonlinearity=activation))
        # followed by Dropout p=0.5
        if dropout:
            layer = DropoutLayer(layer, p=0.5)

        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=0, W=Normal(0.05), nonlinearity=activation))
        # 2x Networks-in-Networks layers with 192 units and lrelu activations
        # We will skip those for now
        # layer = NiN(layer, ...)
        # layer = NiN(layer, ...)

        # 1x Global Pooling Layer
        layer = lasagne.layers.GlobalPoolLayer(layer)

        # 1x Minibatch Discrimination Layer, with 100 kernels
        layer = GAN.MinibatchLayer(layer, num_kernels = 100)

        # 1x Dense layer with 10 units, linear activation, followed by 1x Weight normalization (batch norm) layer
        layer = batch_norm(DenseLayer(layer, num_units = 10, W=Normal(0.05, nonlinearity=None),
                                      train_g=True, init_stdv=0.1))

        # fully-connected layer
        #layer = batch_norm(DenseLayer(layer, 512, nonlinearity=activation))
        # output layer (linear)
        #layer = DenseLayer(layer, 1, nonlinearity=None)

        disc_params = lasagne.layers.get_all_params(layer, trainable=True)
        print ("critic output:", layer.output_shape)
        return layer


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

    def train(self, dataset, num_epochs = 1000, epochsize = 100, batchsize = 64, initial_eta = 1e-4):
        # Load the dataset
        print("Loading data...")

        X_train, X_val, y_train, y_val, ind_train, ind_test = dataset.return_data()

        # Prepare Theano variables for inputs and targets
        noise_var = T.matrix('noise')
        input_var = T.tensor4('inputs')

        # Create neural network model
        print("Building model and compiling functions...")
        generator = build_generator(noise_var)
        critic = build_critic(input_var)

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
        generator_params = lasagne.layers.get_all_params(generator, trainable=True)
        critic_params = lasagne.layers.get_all_params(critic, trainable=True)
        eta = theano.shared(lasagne.utils.floatX(initial_eta))
        generator_updates = lasagne.updates.rmsprop(
            generator_loss, generator_params, learning_rate=eta)
        critic_updates = lasagne.updates.rmsprop(
            critic_loss, critic_params, learning_rate=eta)

        # Instantiate a symbolic noise generator to use for training
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
        noise = srng.uniform((batchsize, 100))

        # Compile functions performing a training step on a mini-batch (according
        # to the updates dictionary) and returning the corresponding score:
        generator_train_fn = theano.function([], generator_loss,
                                             givens={noise_var: noise},
                                             updates=generator_updates)
        critic_train_fn = theano.function([input_var], critic_loss,
                                          givens={noise_var: noise},
                                          updates=critic_updates)

        # Compile another function generating some data
        gen_fn = theano.function([noise_var],
                                 lasagne.layers.get_output(generator,
                                                           deterministic=True))
        ###########################################################################
        ###########################################################################
        ###########################################################################

        # If we have time, add the function to resume training from checkpoints,
        # create regular checkpoints and so on.

        ###########################################################################
        ###########################################################################
        ###########################################################################
        # Finally, launch the training loop.
        print("Starting training...")
        # We create an infinite supply of batches (as an iterable generator):
        batches = iterate_minibatches(X_train, y_train, batchsize, shuffle=True,
                                      forever=True)
        # We iterate over epochs:
        generator_updates = 0
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
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  generator loss: {}".format(np.mean(generator_losses)))
            print("  critic loss:    {}".format(np.mean(critic_losses)))

            # And finally, we plot some generated data
            samples = gen_fn(lasagne.utils.floatX(np.random.rand(42, 100)))
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                pass
            else:
                plt.imsave('lsgan_mnist_samples.png',
                           (samples.reshape(6, 7, 28, 28)
                                   .transpose(0, 2, 1, 3)
                                   .reshape(6*28, 7*28)),
                           cmap='gray')

            # After half the epochs, we start decaying the learn rate towards zero
            if epoch >= num_epochs // 2:
                progress = float(epoch) / num_epochs
                eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

        # Optionally, you could now dump the network weights to a file like this:
        self.generator = generator
        self.discriminator = critic
        self.train_fd = train_fn
        self.generator_fn 
        self.save_model()
        #
        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)

