from __future__ import print_function

import sys, os, time
import numpy as np
import math

import settings
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log

### THINGS TODO:

### Variable definitions
## MOCKING: Right now we are "mocking" the hyper parameters, but layer one we will use the user-provided values
## TODO: Turn these functions into methods of a class which derived from a BaseModel class
# TODO: Once using actual hyper parameters, change the access method so that
# accessing a key which doesn't exist doesn't trigger an unhandled exception, crashing our program.
# Something like that but with a better handling of accessing keys:
# if self.hyper['input_noise']:
#     input_noise = True
# if self.hyper['activation'] == "relu":
#     activation = lasagne.nonlinearities.rectify
# el
#     activation = LeakyRectify(0.2)
# if self.hyper['dropout'] == True:
#     dropout = True


### Utility function "BilinearUpscaleLayer"
### This should be better than using Deconv2DLayer and avoid the artifacts we were getting
### Source: https://github.com/TobyPDE/FRRN/blob/master/dltools/layers.py
import theano.tensor as T
from lasagne.layers import Layer, MergeLayer


class BilinearUpscaleLayer(Layer):
    """
    This layer upscales the 4D input tensor along the trailing spatial dimensions using bilinear interpolation.
    You have to specify image dimensions in order to use this layer - even if you want to have a fully convolutional
    network.
    """
    def __init__(self, incoming, factor, **kwargs):
        """
        Initializes a new instance of the BilinearUpscaleLayer class.
        :param incoming: The incoming network stream
        :param factor: The factor by which to upscale the input
        """
        super(BilinearUpscaleLayer, self).__init__(incoming, **kwargs)
        self.factor = factor

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of the layer given the input shape.
        :param input_shape: The input shape
        :return: The output shape
        """
        return input_shape[0], input_shape[1], self.factor * input_shape[2], self.factor * input_shape[3]

    def get_output_for(self, input, **kwargs):
        """
        Constructs the Theano graph for this layer
        :param input: Symbolic input variable
        :return: Symbolic output variable
        """
        return T.nnet.abstract_conv.bilinear_upsampling(input, self.factor)


################################################################
################################################################
##########         GENERATOR ARCHITECTURES          ############
################################################################
################################################################


def build_generator_architecture(input_var=None, architecture=1):
    ## Module and function imports
    try:
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    except ImportError:
        raise ImportError("Your Lasagne is too old. Try the bleeding-edge "
                          "version: http://lasagne.readthedocs.io/en/latest/"
                          "user/installation.html#bleeding-edge-version")
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        print_warning("Couldn't import batch_norm_dnn from lasagne.layers.dnn. "
                      "Instead, using batch_norm from lasagne.layers.")
        from lasagne.layers import batch_norm
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, DropoutLayer, Conv2DLayer
    from lasagne.nonlinearities import LeakyRectify, sigmoid, tanh
    from lasagne.init import Normal, GlorotUniform, GlorotNormal
    import lasagne.layers as ll
    import gan_lasagne as GAN
    import theano.tensor as T

    layers = []
    def app(l):
        layers.append(l)
        return l
    
    if architecture == -1:
        W_init = Normal(0.05)
        a_fn = LeakyRectify(0.2)
        # input: 100dim
        layer = app(InputLayer(shape=(None, 100), input_var=input_var))
        layer = app(GAN.GaussianNoiseLayer(layer, sigma=0.5))
        layer = app(batch_norm(DenseLayer(layer, 512*4*4)))
        layer = app(ReshapeLayer(layer, ([0], 512, 4, 4)))
        ### four fractional-stride convolutions
        # Note: Apply dropouts in G. See tip #17 from "ganhacks"
        layer = app(batch_norm(Deconv2DLayer(layer, 256, (5, 5), stride=(2, 2), output_size=8, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5))
        layer = app(GAN.GaussianNoiseLayer(layer, sigma=0.2))
        layer = app(batch_norm(Deconv2DLayer(layer, 128, (5, 5), stride=(2, 2), output_size=16, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5))
        layer = GAN.GaussianNoiseLayer(layer, sigma=0.2)
        layer = app(batch_norm(Deconv2DLayer(layer, 64, (5, 5), stride=(2, 2), output_size=32, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5))
        layer = app(GAN.GaussianNoiseLayer(layer, sigma=0.5))
        layer = app(GAN.weight_norm(Deconv2DLayer(layer, 3, (5, 5), stride=(2, 2), output_size=64, nonlinearity=T.tanh), train_g=True, init_stdv=0.1))
        print ("Generator output:", layer.output_shape)
        return layer
    elif architecture == 0:
        a_fn = LeakyRectify(0.2)
        # input: 100dim
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 256*4*4))
        layer = ReshapeLayer(layer, ([0], 256, 4, 4))
        ### four fractional-stride convolutions
        # Note: Apply dropouts in G. See tip #17 from "ganhacks"
        layer = batch_norm(Deconv2DLayer(layer, 256, 7, stride=2, crop='same', output_size=8, nonlinearity=a_fn))
        layer = batch_norm(Deconv2DLayer(layer, 256, 7, stride=1, crop='same', output_size=8, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 192, 5, stride=2, crop='same', output_size=16, nonlinearity=a_fn))
        layer = batch_norm(Deconv2DLayer(layer, 192, 5, stride=1, crop='same', output_size=16, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 128, 3, stride=2, crop='same', output_size=32, nonlinearity=a_fn))
        layer = batch_norm(Deconv2DLayer(layer, 128, 3, stride=1, crop='same', output_size=32, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = Deconv2DLayer(layer, 96, 5, stride=2, crop='same', output_size=64, nonlinearity=a_fn)
        layer = Deconv2DLayer(layer, 3, 5, stride=1, crop='same', output_size=64, nonlinearity=T.tanh)
        print ("Generator output:", layer.output_shape)
        return layer
    elif architecture == 1:
        # input: 100dim
        a_fn = LeakyRectify(0.1)
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 128*8*8))
        layer = ReshapeLayer(layer, ([0], 128, 8, 8))
        ### four fractional-stride convolutions
        # Note: Apply dropouts in G. See tip #17 from "ganhacks"
        layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same',
                                         output_size=16, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(BilinearUpscaleLayer(layer, factor=2)) # output_size=32x32
        layer = Deconv2DLayer(layer, 3, 7, stride=2, crop='same',
                              output_size=64, nonlinearity=T.tanh)
        print ("Generator output:", layer.output_shape)
        return layer
    elif architecture == 2:
        # Optional layers
        input_noise = True
        output_noise = True
        dropout = True

        # Various activation settings
        input_sigma = 0.1 # Gaussian noise to inject to output
        output_sigma = 0.2 # Gaussian noise to inject to output
        alpha = 0.1 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        uniform_range = 0.015
        normal_std = 0.05
        #W_init = Normal(normal_std)
        W_init=GlorotUniform()
        
 
        # Build the network's layers
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # Injecting some noise after input layer
        if input_noise:
            layer = GAN.GaussianNoiseLayer(layer, sigma=input_sigma)
        # fully-connected layer
        # TODO: Do we need this layer???
        #layer = batch_norm(DenseLayer(layer, 1024))
        # project and reshape
        layer = GAN.batch_norm(DenseLayer(layer, num_units = 512*4*4, W=W_init, nonlinearity=a_fn), g=None)
        layer = ReshapeLayer(layer, ([0], 512, 4, 4))
        
        # 3x Deconvs and batch norms
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 256, 8, 8), (5, 5), W=W_init,
                                                 stride=1, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 256, 8, 8), (5, 5), W=W_init,
                                                 stride=2, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer

        # 3x Deconvs and batch norms
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 128, 16, 16), (5, 5), W=W_init,
                                                 stride=1, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 128, 16, 16), (5, 5), W=W_init,
                                                 stride=1, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 128, 16, 16), (5, 5), W=W_init,
                                                 stride=2, nonlinearity=a_fn), g=None)

        # 3x Deconvs and batch norms
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 64, 32, 32), (5, 5), W=W_init,
                                                 stride=1, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 64, 32, 32), (5, 5), W=W_init,
                                                 stride=1, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = GAN.batch_norm(GAN.Deconv2DLayer(layer, (None, 64, 32, 32), (5, 5), W=W_init,
                                                 stride=2, nonlinearity=a_fn), g=None)
        layer = DropoutLayer(layer, p=0.5) if dropout else layer

        # 1x Deconvs and batch norms
        layer = GAN.weight_norm(GAN.Deconv2DLayer(layer, (None, 3, 64, 64), (5, 5), W=W_init,
                                                  stride=1, nonlinearity=T.tanh),
                                train_g=True, init_stdv=0.1)
        
        gen_dat = ll.get_output(layer)
        print ("Generator output:", layer.output_shape)
        return layer

    elif architecture == 3:
        # Optional layers
        input_noise = True
        output_noise = True
        dropout = True

        # Various activation settings
        input_sigma = 0.1 # Gaussian noise to inject to output
        output_sigma = 0.2 # Gaussian noise to inject to output
        alpha = 0.1 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        uniform_range = 0.015
        normal_std = 0.05
        #W_init = Normal(normal_std)
        W_init=GlorotUniform()

        # Build the network's layers
        layer = InputLayer(shape=(None, 100), input_var=input_var)

        # fully-connected layer
        # TODO: Do we need this layer???
        #layer = batch_norm(DenseLayer(layer, 1024))
        # project and reshape
        layer = batch_norm(DenseLayer(layer, num_units = 512*4*4, W=W_init, nonlinearity=a_fn))
        layer = ReshapeLayer(layer, ([0], 512, 4, 4))
        
        # 2x Deconvs and batch norms
        layer = batch_norm(Deconv2DLayer(layer, 256, 5, stride=1, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Deconv2DLayer(layer, 256, 5, stride=2, output_size=8, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer

        # 2x Deconvs and batch norms
        layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=1, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, output_size=16, nonlinearity=a_fn))

        # 2x Deconvs and batch norms
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=1, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=2, output_size=32, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer

        # 2x Deconvs and batch norms
        layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=1, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, output_size=64, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer

        # 1x Deconvs and batch norms
        layer = Deconv2DLayer(layer, 3, 5, stride=1, output_size = 64, nonlinearity=T.tanh)
        
        gen_dat = ll.get_output(layer)
        print ("Generator output:", layer.output_shape)
        return layer
    elif architecture == 4:
        a_fn = LeakyRectify(0.2)
        # input: 100dim
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # Injecting some noise after input layer
        layer = GAN.GaussianNoiseLayer(layer, sigma=0.2)
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 1024))
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 256*4*4))
        layer = ReshapeLayer(layer, ([0], 256, 4, 4))
        ### four fractional-stride convolutions
        # Note: Apply dropouts in G. See tip #17 from "ganhacks"
        layer = batch_norm(Deconv2DLayer(layer, 192, 7, stride=1, crop='same',
                                         output_size=4, nonlinearity=a_fn))
        layer = batch_norm(Deconv2DLayer(layer, 192, 7, stride=2, crop='same',
                                         output_size=8, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 128, 7, stride=1, crop='same',
                                         output_size=8, nonlinearity=a_fn))
        layer = batch_norm(Deconv2DLayer(layer, 128, 7, stride=2, crop='same',
                                         output_size=16, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=1, crop='same',
                                         output_size=16, nonlinearity=a_fn))
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=2, crop='same',
                                         output_size=32, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=1, crop='same',
                                         output_size=32, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = Deconv2DLayer(layer, 3, 5, stride=2, crop='same',
                              output_size=64, nonlinearity=T.tanh)
        print ("Generator output:", layer.output_shape)
        return layer

    elif architecture == 5:
        a_fn = LeakyRectify(0.2)
        W_init = Normal(0.05)

        # input: 100dim
        layer = InputLayer(shape=(None, 100), input_var=input_var)
        # project and reshape
        layer = batch_norm(DenseLayer(layer, 512*4*4, W=W_init, nonlinearity=a_fn))
        layer = ReshapeLayer(layer, ([0], 512, 4, 4))
        ### four fractional-stride convolutions
        # Note: Apply dropouts in G. See tip #17 from "ganhacks"
        layer = batch_norm(Deconv2DLayer(layer, 256, 5, stride=2, crop='same', W=W_init,
                                         output_size=8, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 128, 5, stride=2, crop='same', W=W_init,
                                         output_size=16, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Deconv2DLayer(layer, 96, 5, stride=2, crop='same', W=W_init,
                                         output_size=32, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = Deconv2DLayer(layer, 3, 5, stride=2, crop='same', W=W_init,
                              output_size=64, nonlinearity=T.tanh)
        print ("Generator output:", layer.output_shape)
        return layer
    elif architecture == 6 or architecture == 7:
        layer = app(InputLayer(shape=(None, 100), input_var=input_var))
        layer = app(GAN.batch_norm(ll.DenseLayer(layer, num_units=4*4*512, W=Normal(0.05), nonlinearity=GAN.relu), g=None))
        layer = app(ll.ReshapeLayer(layer, ([0],512,4,4)))
        layer = app(GAN.batch_norm(GAN.Deconv2DLayer(layer, (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=GAN.relu), g=None)) # 4 -> 8
        layer = app(GAN.batch_norm(GAN.Deconv2DLayer(layer, (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=GAN.relu), g=None)) # 8 -> 16
        layer = app(GAN.batch_norm(GAN.Deconv2DLayer(layer, (None,64,32,32), (5,5), W=Normal(0.05), nonlinearity=GAN.relu), g=None)) # 16 -> 32
        layer = app(GAN.weight_norm(GAN.Deconv2DLayer(layer, (None,3,64,64), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 32 -> 64

        gen_dat = ll.get_output(layer)

        print ("Generator output:", layer.output_shape)
        return layer

    raise Exception("Invalid argument to LSGAN's build_generator: architecture = {}".format(architecture))
    

################################################################
################################################################
#############      CRITIC ARCHITECTURES           ##############
################################################################
################################################################


def build_critic_architecture(input_var=None, architecture=1):
    ## Module and function imports
    try:
        from lasagne.layers.dnn import batch_norm_dnn as batch_norm
    except ImportError:
        print_warning("Couldn't import batch_norm_dnn from lasagne.layers.dnn. "
                      "Instead, using batch_norm from lasagne.layers.")
        from lasagne.layers import batch_norm
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, DropoutLayer
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer, GaussianNoiseLayer,
                                DenseLayer, NINLayer, DropoutLayer, MaxPool2DLayer)
    from lasagne.nonlinearities import LeakyRectify, sigmoid, tanh
    from lasagne.init import Normal, GlorotUniform, GlorotNormal
    import lasagne.layers as ll
    import gan_lasagne as GAN
    import theano.tensor as T

    layers = []
    def app(l):
        layers.append(l)
        return l

    if architecture == -1:
        try:
            from lasagne.layers import dnn
        except ImportError as e:
            raise ImportError("Architecture #0 of LSGAN requires lasagne.layers.dnn (which requires "
                              "a functional cuDNN installation).")
    elif architecture == 0:
        alpha = 0.2 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        # input: (None, 3, 64, 64)
        layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
        # Injecting some noise after input layer
        layer = GaussianNoiseLayer(layer, sigma=0.1)
        # four convolutions
        # convolutions
        layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same', nonlinearity=a_fn))
        layer = MaxPool2DLayer(layer, 2)
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same', nonlinearity=a_fn))
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 512, nonlinearity=a_fn))
        # Minibatch layer
        layer = GAN.MinibatchLayer(layer, num_kernels = 150, dim_per_kernel=5, theta=Normal(0.05))
        # output layer (linear)
        layer = DenseLayer(layer, 1, nonlinearity=None)
        print ("critic output:", layer.output_shape)
        return layer
    elif architecture == 1:
        alpha = 0.2 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        # input: (None, 3, 64, 64)
        layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
        # Injecting some noise after input layer
        layer = GaussianNoiseLayer(layer, sigma=0.2)
        # convolutions
        layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same', nonlinearity=a_fn))
        layer = MaxPool2DLayer(layer, 2)
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same', nonlinearity=a_fn))
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 512, nonlinearity=a_fn))
        # output layer (linear)
        layer = DenseLayer(layer, 1, nonlinearity=None)
        print ("critic output:", layer.output_shape)
        return layer
    elif architecture == 2:
        alpha = 0.2 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        # input: (None, 3, 64, 64)
        layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
        # Injecting some noise after input layer
        layer = GaussianNoiseLayer(layer, sigma=0.1)
        # convolutions
        layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same', nonlinearity=a_fn))
        layer = MaxPool2DLayer(layer, 2)
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same', nonlinearity=a_fn))
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 512, nonlinearity=a_fn))
        # NiN layer
        layer = batch_norm(NINLayer(layer, num_units=64, nonlinearity=a_fn))
        # Global pooling
        layer = ll.GlobalPoolLayer(layer)
        # Minibatch layer
        layer = GAN.MinibatchLayer(layer, num_kernels = 150, dim_per_kernel=5, theta=Normal(0.05))
        # output layer (linear)
        layer = DenseLayer(layer, 1, nonlinearity=None)
        print ("critic output:", layer.output_shape)
        return layer
    elif architecture == 3:
        # Optional layers
        input_noise = True
        output_noise = True
        dropout = True

        # Various activation settings
        input_sigma = 0.1 # Gaussian noise to inject to output
        output_sigma = 0.2 # Gaussian noise to inject to output
        alpha = 0.25 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        uniform_range = 0.015
        normal_std = 0.05
        W_init = Normal(normal_std)

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
        #if input_noise:
        #    layer = GAN.GaussianNoiseLayer(layer, sigma=input_sigma)
        layer = DropoutLayer(layer, p=0.1) if dropout else layer

        # 3x convolutions with 96 filters each and 3x3 receptive field
        # 1st, 2nd conv preserve dimension
        # 3rd conv has stride 2, so it downsamples dimension by a factor of 2
        layer = batch_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        # This layer turns the feature maps into tensors: (None, 96, 32, 32)
        layer = batch_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, stride=2, W=W_init, nonlinearity=a_fn))
            
        # 3x convolutions with 192 filters each and 3x3 receptive field
        # 1st, 2nd conv preserve dimension
        # 3rd conv has stride 2, so it downsamples dimension by a factor of 2
        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        # This layer turns the feature maps into tensors: (None, 192, 16, 16)
        layer = batch_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, stride=2, W=W_init, nonlinearity=a_fn))

        layer = batch_norm(Conv2DLayer(layer, 256, (3, 3), pad=1, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        layer = batch_norm(Conv2DLayer(layer, 256, (3, 3), pad=1, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        # This layer turns the feature maps into tensors: (None, 256, 8, 8)
        layer = batch_norm(Conv2DLayer(layer, 256, (3, 3), pad=1, stride=2, W=W_init, nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5) if dropout else layer
        
        layer = batch_norm(Conv2DLayer(layer, 256, (3, 3), pad=0, W=W_init, nonlinearity=a_fn))
        # 2x Networks-in-Networks layers with 192 units and lrelu activations
        # We will skip those for now
        layer = batch_norm(NINLayer(layer, num_units=256, W=W_init, nonlinearity=a_fn))
        layer = batch_norm(NINLayer(layer, num_units=256, W=W_init, nonlinearity=a_fn))

        # fully-connected layer
        #layer = weight_norm(DenseLayer(layer, 512, nonlinearity=a_fn))

        # 1x Global Pooling Layer
        layer = ll.GlobalPoolLayer(layer)

        # 1x Minibatch Discrimination Layer, with 250 kernels
        #layer = GAN.MinibatchLayer(layer, num_kernels = 250, dim_per_kernel=5, theta=W_init)

        # Apply Gaussian noise to output
        #if output_noise:
        #    layer = GAN.GaussianNoiseLayer(layer, sigma=output_sigma)

        # 1x Dense layer with 1 units, linear activation, followed by 1x Weight normalization (batch norm) layer
        layer = batch_norm(DenseLayer(layer, num_units = 1, W=W_init, nonlinearity=None))

        disc_params = ll.get_all_params(layer, trainable=True)
        print ("critic output:", layer.output_shape)
        return layer

    elif architecture == 4:
        # Optional layers
        input_noise = True
        output_noise = True
        dropout = True

        # Various activation settings
        input_sigma = 0.1 # Gaussian noise to inject to output
        output_sigma = 0.2 # Gaussian noise to inject to output
        alpha = 0.1 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        uniform_range = 0.015
        normal_std = 0.05
        W_init = Normal(normal_std)

        # input: (None, 3, 64, 64)
        layer = InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
        # Injecting some noise after input layer
        if input_noise:
            layer = GAN.GaussianNoiseLayer(layer, sigma=0.2)

        # four convolutions
        layer = batch_norm(Conv2DLayer(layer, 96, 5, stride=1, pad='same',
                                       nonlinearity=a_fn))
        layer = batch_norm(Conv2DLayer(layer, 96, 5, stride=2, pad='same',
                                       nonlinearity=a_fn))
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=1, pad='same',
                                       nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                       nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Conv2DLayer(layer, 192, 7, stride=1, pad='same',
                                       nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Conv2DLayer(layer, 192, 7, stride=2, pad='same',
                                       nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Conv2DLayer(layer, 256, 7, stride=1, pad='same',
                                       nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        layer = batch_norm(Conv2DLayer(layer, 256, 7, stride=2, pad='same',
                                       nonlinearity=a_fn))
        layer = DropoutLayer(layer, p=0.5)
        # fully-connected layer
        layer = batch_norm(DenseLayer(layer, 512, nonlinearity=a_fn))

        # Apply Gaussian noise to output
        if output_noise:
            layer = GAN.GaussianNoiseLayer(layer, sigma=output_sigma)

        # output layer (linear)
        layer = DenseLayer(layer, 1, nonlinearity=None)
        print ("critic output:", layer.output_shape)
        return layer
    elif architecture == 5:
        # Optional layers
        input_noise = True
        output_noise = True
        dropout = True

        # Various activation settings
        input_sigma = 0.1 # Gaussian noise to inject to output
        output_sigma = 0.2 # Gaussian noise to inject to output
        alpha = 0.25 # slope of negative x axis of leaky ReLU
        a_fn = LeakyRectify(alpha)
        uniform_range = 0.015
        normal_std = 0.05
        W_init = Normal(normal_std)

        # input: (None, 3, 64, 64)
        layer = app(InputLayer(shape=(None, 3, 64, 64), input_var=input_var))
        # new layer: What is it for???
        # ganhacks says:
        # * Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
        #    - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        #    - https://openreview.net/forum?id=Hk4_qw5xe
        # * Adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
        #    - (Zhao et. al. EBGAN)
        #    - "Improved GANs" by OpenAI. Their code also has it, but is commented out
        if input_noise:
            layer = app(GAN.GaussianNoiseLayer(layer, sigma=input_sigma))
        layer = app(DropoutLayer(layer, p=0.1)) if dropout else layer

        # 3x convolutions with 96 filters each and 3x3 receptive field
        # 1st, 2nd conv preserve dimension
        # 3rd conv has stride 2, so it downsamples dimension by a factor of 2
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else laye
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else layer
        # This layer turns the feature maps into tensors: (None, 96, 32, 32)
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 96, (3, 3), pad=1, stride=2, W=W_init, nonlinearity=a_fn)))
            
        # 3x convolutions with 192 filters each and 3x3 receptive field
        # 1st, 2nd conv preserve dimension
        # 3rd conv has stride 2, so it downsamples dimension by a factor of 2
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else layer
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else layer
        # This layer turns the feature maps into tensors: (None, 192, 16, 16)
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 192, (3, 3), pad=1, stride=2, W=W_init, nonlinearity=a_fn)))

        layer = app(GAN.weight_norm(Conv2DLayer(layer, 256, (3, 3), pad=1, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else layer
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 256, (3, 3), pad=1, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else layer
        # This layer turns the feature maps into tensors: (None, 256, 8, 8)
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 256, (3, 3), pad=1, stride=2, W=W_init, nonlinearity=a_fn)))
        layer = app(DropoutLayer(layer, p=0.5)) if dropout else layer
        
        layer = app(GAN.weight_norm(Conv2DLayer(layer, 256, (3, 3), pad=0, W=W_init, nonlinearity=a_fn)))
        # 2x Networks-in-Networks layers with 192 units and lrelu a_fns
        # We will skip those for now
        layer = app(GAN.weight_norm(NINLayer(layer, num_units=256, W=W_init, nonlinearity=a_fn)))
        layer = app(GAN.weight_norm(NINLayer(layer, num_units=256, W=W_init, nonlinearity=a_fn)))

        # fully-connected layer
        #layer = app(weight_norm(DenseLayer(layer, 512, nonlinearity=a_fn)))

        # 1x Global Pooling Layer
        layer = app(ll.GlobalPoolLayer(layer))

        # 1x Minibatch Discrimination Layer, with 250 kernels
        layer = app(GAN.MinibatchLayer(layer, num_kernels = 250, dim_per_kernel=5, theta=W_init))

        # Apply Gaussian noise to output
        if output_noise:
            layer = app(GAN.GaussianNoiseLayer(layer, sigma=output_sigma))

        # 1x Dense layer with 1 units, linear activation, followed by 1x Weight normalization (batch norm) layer
        layer = app(GAN.weight_norm(DenseLayer(layer, num_units = 1, W=W_init, nonlinearity=None),
                                train_g=True, init_stdv=0.1))

        disc_params = ll.get_all_params(layer, trainable=True)
        print ("critic output:", layer.output_shape)
        return layer
    elif architecture == 6:
        try:
            from lasagne.layers import dnn
        except ImportError as e:
            raise ImportError("Architecture #6 of LSGAN requires lasagne.layers.dnn (which requires "
                              "a functional cuDNN installation).")

        layer = (InputLayer(shape=(None, 3, 64, 64), input_var=input_var))
        layer = (DropoutLayer(layer, p=0.2))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.DropoutLayer(layer, p=0.5))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.DropoutLayer(layer, p=0.5))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 256, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.DropoutLayer(layer, p=0.5))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(ll.NINLayer(layer, num_units=192, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(ll.NINLayer(layer, num_units=192, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.GlobalPoolLayer(layer))
        layer = (GAN.weight_norm(ll.DenseLayer(layer, num_units=1, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
        disc_params = ll.get_all_params(layer, trainable=True)

        print ("critic output:", layer.output_shape)
        return layer
    elif architecture == 7:
        try:
            from lasagne.layers import dnn
        except ImportError as e:
            raise ImportError("Architecture #6 of LSGAN requires lasagne.layers.dnn (which requires "
                              "a functional cuDNN installation).")

        layer = (InputLayer(shape=(None, 3, 64, 64), input_var=input_var))
        layer = (GAN.GaussianNoiseLayer(layer, sigma=0.2))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.DropoutLayer(layer, p=0.5))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.DropoutLayer(layer, p=0.5))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 256, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.DropoutLayer(layer, p=0.5))
        layer = (GAN.weight_norm(dnn.Conv2DDNNLayer(layer, 256, (3,3), pad=1, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(ll.NINLayer(layer, num_units=192, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (GAN.weight_norm(ll.NINLayer(layer, num_units=192, W=Normal(0.05), nonlinearity=GAN.lrelu)))
        layer = (ll.GlobalPoolLayer(layer))
        layer = (GAN.MinibatchLayer(layer, num_kernels=100))
        layer = (GAN.weight_norm(ll.DenseLayer(layer, num_units=1, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))

        disc_params = ll.get_all_params(layer, trainable=True)

        print ("critic output:", layer.output_shape)
        return layer
    
    raise Exception("Invalid argument to LSGAN's build_critic: architecture = {}".format(architecture))

