from __future__ import print_function
# coding: utf-8
# -*- coding: utf-8 -*-

"""
CREDIT: Code taken from "f0k/dcgan_mnist.py" on github (https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56) and modified to handle the project's dataset.

---

Example employing Lasagne for digit generation using the MNIST dataset and
Deep Convolutional Generative Adversarial Networks
(DCGANs, see http://arxiv.org/abs/1511.06434).

It is based on the MNIST example in Lasagne:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

Note: In contrast to the original paper, this trains the generator and
discriminator at once, not alternatingly. It's easy to change, though.

Jan Schluter, 2015-12-16
"""

#input_dim=(3, 64, 64)
#output_dim=(3, 32, 32)

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

# ##################### Build the neural network model #######################
# We create two models: The generator and the discriminator network. The
# generator needs a transposed convolution layer defined first.

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)


def build_generator(input_var=None):
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
    from lasagne.nonlinearities import sigmoid
    # input: 100dim
    layer = InputLayer(shape=(None, 100), input_var=input_var)
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*4*4))
    layer = ReshapeLayer(layer, ([0], 128, 4, 4))
    # four fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    layer = batch_norm(Deconv2DLayer(layer, 32, 5, stride=2, pad=2))
    layer = batch_norm(Deconv2DLayer(layer, 16, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, 3, 5, stride=2, pad=2,
                          nonlinearity=sigmoid)
    #layer = ReshapeLayer(layer, (None, 3, 64, 64))
    print ("Generator output:", layer.output_shape)
    return layer


def build_discriminator(input_var=None):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm)
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 3, 64, 64)
    layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
    # four convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 256, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 512, 5, stride=2, pad=2, nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=sigmoid)
    print ("Discriminator output:", layer.output_shape)
    return layer
    

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(Dataset, num_epochs=200, batchsize=128, initial_eta=2e-4):
    # Load the dataset
    print("Loading data...")
    X_train, X_test, y_train, y_test, ind_train, ind_test = Dataset.return_data()

    # Prepare Theano variables for inputs and targets
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
#    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    generator = build_generator(noise_var)
    discriminator = build_discriminator(input_var)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator,
                                         lasagne.layers.get_output(generator))

    # Create loss expressions
    generator_loss = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1)
            + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()

    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=eta, beta1=0.5)
    updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=eta, beta1=0.5))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([noise_var, input_var],
                               [(real_out > .5).mean(),
                                (fake_out < .5).mean()],
                               updates=updates)

    # Compile another function generating some data
    gen_fn = theano.function([noise_var],
                             lasagne.layers.get_output(generator,
                                                       deterministic=True))

    # Create experiment's results directories
    settings.touch_dir(settings.MODELS_DIR)
    settings.touch_dir(settings.EPOCHS_DIR)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            noise = lasagne.utils.floatX(np.random.rand(len(inputs), 100))
            train_err += np.array(train_fn(noise, inputs))
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{}".format(train_err / train_batches))

        # And finally, we plot some generated data
        samples = gen_fn(lasagne.utils.floatX(np.random.rand(10*10, 100)))
        sample = gen_fn(lasagne.utils.floatX(np.random.rand(1, 100)))

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
    np.savez(os.path.join(settings.MODELS_DIR, 'dcgan_gen.npz'), *lasagne.layers.get_all_param_values(generator))
    np.savez(os.path.join(settings.MODELS_DIR, 'dcgan_disc.npz'), *lasagne.layers.get_all_param_values(discriminator))

    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return generator, discriminator, train_fn, gen_fn


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a DCGAN on MNIST using Lasagne.")
        print("Usage: %s [EPOCHS]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
