# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
import settings
import numpy as np
from lasagne_models import LasagneModel, Lasagne_Conv_Deconv
import hyper_params
from utils import print_critical, print_error, print_warning, print_info, print_positive, log, logout

    
class VGG16_Model(LasagneModel):
    def __init__(self, hyperparams = hyper_params.default_vgg16_hyper_params):
        super(VGG16_Model, self).__init__(hyperparams = hyperparams)
        self.input_prevgg = None
        self.input_prevgg_out = None
        self.target_prevgg = None
        self.target_prevgg_out = None
        self.input_vgg_model = None
        self.input_vgg_model_out = None
        self.target_vgg_model = None
        self.target_vgg_model_out = None

    def build(self):
        pass
        
    def build_network_and_loss(self, input_var, target_var):
        import lasagne
        from lasagne.layers import InputLayer
        from lasagne.layers import DenseLayer
        from lasagne.layers import NonlinearityLayer
        from lasagne.layers import DropoutLayer
        from lasagne.layers import Pool2DLayer as PoolLayer
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        from lasagne.nonlinearities import softmax, sigmoid, tanh
        import cPickle as pickle

        try:
            from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
        except ImportError as e:
            from lasagne.layers import Conv2DLayer as ConvLayer
            print_warning("Cannot import 'lasagne.layers.dnn.Conv2DDNNLayer' as it requires GPU support and a functional cuDNN installation. Falling back on slower convolution function 'lasagne.layers.Conv2DLayer'.")

        batch_size = settings.BATCH_SIZE

        net = {}

        net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        net['conv1'] = ConvLayer(net['input'], 256, 5, stride=2, pad='same') # 32x32
        net['conv2'] = ConvLayer(net['conv1'], 256, 7, stride=2, pad='same') # 16x16
        net['dropout1'] = DropoutLayer(net['conv2'], p=0.5)
        net['deconv1'] = Deconv2DLayer(net['dropout1'], 256, 7, stride=1, crop='same', output_size=8) # 16x16
        net['dropout2'] = DropoutLayer(net['deconv1'], p=0.5)
        net['deconv2'] = Deconv2DLayer(net['dropout2'], 256, 7, stride=2, crop='same', output_size=16) # 32x32
        net['dropout3'] = DropoutLayer(net['deconv2'], p=0.5)
        net['deconv3'] = Deconv2DLayer(net['dropout3'], 256, 9, stride=1, crop='same', output_size=32) # 32x32
        net['deconv4'] = Deconv2DLayer(net['deconv3'], 3, 9, stride=1, crop='same', output_size=32, nonlinearity=tanh)

        self.network, self.network_out = net, net['deconv4']
        self.input_pad, self.input_pad_out = self.build_pad_model(self.network_out)
        self.target_pad, self.target_pad_out = self.build_pad_model(InputLayer((batch_size, 3, 32, 32), input_var=target_var))

        self.input_vgg_model, self.input_vgg_model_out = self.build_vgg_model(self.input_pad_out)
        self.target_vgg_model, self.target_vgg_model_out = self.build_vgg_model(self.target_pad_out)

    def build_train_loss(self, input_var, target_var):
        from lasagne.layers import get_output, squared_error
        import theano
        import theano.tensor as T

        loss_conv_1_1 = squared_error(get_output(self.input_vgg_model['conv1_1']),
                                      get_output(self.target_vgg_model['conv1_1'])).mean()
        loss_conv_2_1 = squared_error(get_output(self.input_vgg_model['conv2_1']),
                                      get_output(self.target_vgg_model['conv2_1'])).mean()
        loss_conv_3_1 = squared_error(get_output(self.input_vgg_model['conv3_1']),
                                      get_output(self.target_vgg_model['conv3_1'])).mean()

        return loss_conv_1_1 + loss_conv_2_1 + loss_conv_3_1

    def build_test_loss(self, input_var, target_var):
        from lasagne.layers import get_output, squared_error
        import theano
        import theano.tensor as T

        loss_conv_1_1 = squared_error(get_output(self.input_vgg_model['conv1_1'], deterministic=True),
                                      get_output(self.target_vgg_model['conv1_1'], deterministic=True)).mean()
        loss_conv_2_1 = squared_error(get_output(self.input_vgg_model['conv2_1'], deterministic=True),
                                      get_output(self.target_vgg_model['conv2_1'], deterministic=True)).mean()
        loss_conv_3_1 = squared_error(get_output(self.input_vgg_model['conv3_1'], deterministic=True),
                                      get_output(self.target_vgg_model['conv3_1'], deterministic=True)).mean()

        return loss_conv_1_1 + loss_conv_2_1 + loss_conv_3_1
    
    def build_pad_model(self, previous_layer):
        from lasagne.layers import InputLayer
        from lasagne.layers import PadLayer

        padnet = {}
        padnet['input'] = previous_layer
        padnet['pad'] = PadLayer(padnet['input'], (224-32)/2)
        return padnet, padnet['pad']
        
    def build_vgg_model(previous_layer):
        log("Building VGG-16 model...")

        net = {}
        net['input'] = previous_layer
        net['conv1_1'] = ConvLayer(
            net['input'], 64, 3, pad=1, flip_filters=False)
        net['conv1_2'] = ConvLayer(
            net['conv1_1'], 64, 3, pad=1, flip_filters=False)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(
            net['pool1'], 128, 3, pad=1, flip_filters=False)
        net['conv2_2'] = ConvLayer(
            net['conv2_1'], 128, 3, pad=1, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(
            net['pool2'], 256, 3, pad=1, flip_filters=False)
        net['conv3_2'] = ConvLayer(
            net['conv3_1'], 256, 3, pad=1, flip_filters=False)
        net['conv3_3'] = ConvLayer(
            net['conv3_2'], 256, 3, pad=1, flip_filters=False)
        net['pool3'] = PoolLayer(net['conv3_3'], 2)
        net['conv4_1'] = ConvLayer(
            net['pool3'], 512, 3, pad=1, flip_filters=False)
        net['conv4_2'] = ConvLayer(
            net['conv4_1'], 512, 3, pad=1, flip_filters=False)
        net['conv4_3'] = ConvLayer(
            net['conv4_2'], 512, 3, pad=1, flip_filters=False)
        net['pool4'] = PoolLayer(net['conv4_3'], 2)
        net['conv5_1'] = ConvLayer(
            net['pool4'], 512, 3, pad=1, flip_filters=False)
        net['conv5_2'] = ConvLayer(
            net['conv5_1'], 512, 3, pad=1, flip_filters=False)
        net['conv5_3'] = ConvLayer(
            net['conv5_2'], 512, 3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5_3'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
        net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(
            net['fc7_dropout'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)
        net_output = net['prob']

        log("Loading VGG16 pre-trained weights from file 'vgg16.pkl'...")
        with open('vgg16.pkl', 'rb') as f:
            params = pickle.load(f)

        #net_output.initialize_layers()
        lasagne.layers.set_all_param_values(net['prob'], params['param values'])

        return net, net['prob']
