#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example employing Lasagne for digit generation using the MNIST dataset and
Wasserstein Generative Adversarial Networks
(WGANs, see https://arxiv.org/abs/1701.07875 for the paper and
https://github.com/martinarjovsky/WassersteinGAN for the "official" code).

It is based on a DCGAN example:
https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
This, in turn, is based on the MNIST example in Lasagne:
https://lasagne.readthedocs.io/en/latest/user/tutorial.html

Jan Schlüter, 2017-02-02
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import PIL.Image as Image

import lasagne
import settings
import dataset
from utils import normalize_data, denormalize_data 
from utils import log

# ##################### Build the neural network model #######################
# We create two models: The generator and the critic network.
# The models are the same as in the Lasagne DCGAN example, except that the
# discriminator is now a critic with linear output instead of sigmoid output.

def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
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
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*4*4))
    layer = ReshapeLayer(layer, ([0], 128, 4, 4))
    # four fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=8))
    layer = batch_norm(Deconv2DLayer(layer, 32, 5, stride=2, crop='same',
                                     output_size=16))
    layer = batch_norm(Deconv2DLayer(layer, 16, 5, stride=2, crop='same',
                                     output_size=32))
    layer = Deconv2DLayer(layer, 3, 5, stride=2, crop='same', output_size=64,
                          nonlinearity=sigmoid)
    print ("Generator output:", layer.output_shape)
    return layer

def build_critic(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer)
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify
    lrelu = LeakyRectify(0.2)
    # input: (None, 3, 64, 64)
    layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 256, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 512, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer (linear and without bias)
    layer = DenseLayer(layer, 1, nonlinearity=None, b=None)
    print ("critic output:", layer.output_shape)
    return layer
    

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False,
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

def train(Dataset, num_epochs=1000, epochsize=100, batchsize=64, initial_eta=5e-5,
         clip=0.01):
    # Load the dataset
    log("Loading data...")
    X_train, X_test, y_train, y_test, ind_train, ind_test = Dataset.return_data()

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')

    # Create neural network model
    log("Building model and compiling functions...")
    generator = build_generator(noise_var)
    critic = build_critic(input_var)

    # Create expression for passing real data through the critic
    real_out = lasagne.layers.get_output(critic)
    # Create expression for passing fake data through the critic
    fake_out = lasagne.layers.get_output(critic,
            lasagne.layers.get_output(generator))
    
    # Create score expressions to be maximized (i.e., negative losses)
    generator_score = fake_out.mean()
    critic_score = real_out.mean() - fake_out.mean()
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    critic_params = lasagne.layers.get_all_params(critic, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    generator_updates = lasagne.updates.rmsprop(
            -generator_score, generator_params, learning_rate=eta)
    critic_updates = lasagne.updates.rmsprop(
            -critic_score, critic_params, learning_rate=eta)

    # Clip critic parameters in a limited range around zero (except biases)
    for param in lasagne.layers.get_all_params(critic, trainable=True,
                                               regularizable=True):
        critic_updates[param] = T.clip(critic_updates[param], -clip, clip)

    # Instantiate a symbolic noise generator to use for training
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(seed=np.random.randint(2147462579, size=6))
    noise = srng.uniform((batchsize, 100))

    # Compile functions performing a training step on a mini-batch (according
    # to the updates dictionary) and returning the corresponding score:
    generator_train_fn = theano.function([], generator_score,
                                         givens={noise_var: noise},
                                         updates=generator_updates)
    critic_train_fn = theano.function([input_var], critic_score,
                                      givens={noise_var: noise},
                                      updates=critic_updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Create experiment's results directories
    settings.touch_dir(settings.MODELS_DIR)
    settings.touch_dir(settings.EPOCHS_DIR)

    # Finally, launch the training loop.
    log("Starting training...")
    # We create an infinite supply of batches (as an iterable generator):
    batches = iterate_minibatches(X_train, y_train, batchsize, shuffle=True,
                                  forever=True)
    # We iterate over epochs:
    generator_updates = 0
    for epoch in range(num_epochs):
        start_time = time.time()

        # In each epoch, we do `epochsize` generator updates. Usually, the
        # critic is updated 5 times before every generator update. For the
        # first 25 generator updates and every 500 generator updates, the
        # critic is updated 100 times instead, following the authors' code.
        critic_scores = []
        generator_scores = []
        for _ in range(epochsize):
            if (generator_updates < 25) or (generator_updates % 500 == 0):
                critic_runs = 100
            else:
                critic_runs = 5
            for _ in range(critic_runs):
                batch = next(batches)
                inputs, targets = batch
                critic_scores.append(critic_train_fn(inputs))
            generator_scores.append(generator_train_fn())
            generator_updates += 1

        # Then we print the results for this epoch:
        log("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        log("  generator score:\t\t{}".format(np.mean(generator_scores)))
        log("  Wasserstein distance:\t\t{}".format(np.mean(critic_scores)))

        # And finally, we plot some generated data
        samples = np.array(gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100))))
        sample = np.array(gen_fn(lasagne.utils.floatX(np.random.rand(1, 100))))

        samples = dataset.denormalize_data(samples)
        sample = dataset.denormalize_data(sample)

        samples_path = os.path.join(settings.EPOCHS_DIR, 'samples_epoch%i.png' % (epoch + 1))
        Image.fromarray(samples.reshape(10, 10, 3, 64, 64)
                        .transpose(0, 3, 1, 4, 2)
                        .reshape(10*64, 10*64, 3)).save(samples_path)

        sample_path = os.path.join(settings.EPOCHS_DIR, 'one_sample_epoch%i.png' % (epoch + 1))
        Image.fromarray(sample.reshape(3, 64, 64).transpose(1, 2, 0).reshape(64, 64, 3)).save(sample_path)

        # After half the epochs, we start decaying the learn rate towards zero
        if epoch >= num_epochs // 2:
            progress = float(epoch) / num_epochs
            eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez(os.path.join(settings.MODELS_DIR, 'wgan_gen.npz'), *lasagne.layers.get_all_param_values(generator))
    np.savez(os.path.join(settings.MODELS_DIR, 'wgan_crit.npz'), *lasagne.layers.get_all_param_values(critic))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return generator, critic, generator_train_fn, critic_train_fn, gen_fn
