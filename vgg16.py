# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
import numpy as np
import theano
import theano.tensor as T
import lasagne

from lasagne_models import LasagneModel, Lasagne_Conv_Deconv

import settings
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
        
    def build_network(self, input_var, target_var):
        from lasagne.layers import InputLayer
        from lasagne.layers import DenseLayer
        from lasagne.layers import NonlinearityLayer
        from lasagne.layers import DropoutLayer
        from lasagne.layers import ReshapeLayer
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
        net['conv1'] = ConvLayer(net['input'], 512, 5, stride=2, pad='same') # 32x32
        net['conv2'] = ConvLayer(net['conv1'], 256, 7, stride=1, pad='same') # 32x32
        net['dropout1'] = DropoutLayer(net['conv2'], p=0.5)
        net['deconv1'] = Deconv2DLayer(net['dropout1'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        net['dropout2'] = DropoutLayer(net['deconv1'], p=0.5)
        net['deconv2'] = Deconv2DLayer(net['dropout2'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        net['dropout3'] = DropoutLayer(net['deconv2'], p=0.5)
        net['deconv3'] = Deconv2DLayer(net['dropout3'], 256, 9, stride=1, crop='same', output_size=32) # 32x32
        net['dropout4'] = DropoutLayer(net['deconv3'], p=0.5)
        net['fc1'] = DenseLayer(net['dropout4'], 1024)
        net['dropout5'] = DropoutLayer(net['fc1'], p=0.5)
        net['fc2'] = DenseLayer(net['dropout5'], 3*32*32)
        net['output'] = net['reshape'] = ReshapeLayer(net['fc2'], ([0], 3, 32, 32))
        
        # net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        # net['dropout1'] = DropoutLayer(net['input'], p=0.1)
        # net['conv1'] = ConvLayer(net['dropout1'], 256, 5, stride=2, pad='same') # 32x32
        # net['dropout2'] = DropoutLayer(net['conv1'], p=0.5)
        # net['conv2'] = ConvLayer(net['dropout2'], 256, 7, stride=1, pad='same') # 32x32
        # net['dropout3'] = DropoutLayer(net['conv2'], p=0.5)
        # net['deconv1'] = Deconv2DLayer(net['dropout3'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        # net['dropout4'] = DropoutLayer(net['deconv1'], p=0.5)
        # net['deconv3'] = Deconv2DLayer(net['dropout4'], 256, 9, stride=1, crop='same', output_size=32) # 32x32
        # net['dropout5'] = DropoutLayer(net['deconv3'], p=0.5)
        # net['fc1'] = DenseLayer(net['dropout5'], 2048)
        # net['dropout6'] = DropoutLayer(net['fc1'], p=0.5)
        # net['fc2'] = DenseLayer(net['dropout6'], 2048)
        # net['dropout7'] = DropoutLayer(net['fc2'], p=0.5)
        # net['fc3'] = DenseLayer(net['dropout7'], 3*32*32)
        # net['dropout8'] = DropoutLayer(net['fc3'], p=0.5)
        # net['reshape'] = ReshapeLayer(net['dropout8'], ([0], 3, 32, 32))
        # net['output'] = Deconv2DLayer(net['reshape'], 3, 9, stride=1, crop='same', output_size=32, nonlinearity=sigmoid)

        self.network, self.network_out = net, net['output']
        print ("Conv_Deconv network output shape:   {}".format(self.network_out.output_shape))
        # self.input_pad, self.input_pad_out = self.build_pad_model(self.network_out)
        # self.target_pad, self.target_pad_out = self.build_pad_model(InputLayer((batch_size, 3, 32, 32), input_var=target_var))
        self.input_scaled, self.input_scaled_out = self.build_scaled_model(self.network_out)
        self.target_scaled, self.target_scaled_out = self.build_scaled_model(InputLayer((batch_size, 3, 32, 32), input_var=target_var))
        print("(Input) scaled network output shape:  {}".format(self.input_scaled_out.output_shape))
        print("(Target) scaled network output shape: {}".format(self.target_scaled_out.output_shape))

        self.vgg_scaled_var = T.tensor4('scaled_vars')
        
        self.vgg_model, self.vgg_model_out = self.build_vgg_model(self.vgg_scaled_var)
        print("VGG model conv1_1 output shape: {}".format(self.vgg_model['conv1_1'].output_shape))
        print("VGG model conv2_1 output shape: {}".format(self.vgg_model['conv2_1'].output_shape))
        print("VGG model conv3_1 output shape: {}".format(self.vgg_model['conv3_1'].output_shape))

    def build_pad_model(self, previous_layer):
        from lasagne.layers import PadLayer
        padnet = {}
        padnet['input'] = previous_layer
        padnet['pad'] = PadLayer(padnet['input'], (224-32)/2)
        return padnet, padnet['pad']

    def build_scaled_model(self, previous_layer):
        from lasagne.layers import TransformerLayer

        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 7.0
        b[1, 1] = 7.0
        b = b.flatten()  # identity transform
        W = lasagne.init.Constant(0.0)
        scalenet = {}
        scalenet['input'] = previous_layer
        scalenet['scale_init'] = lasagne.layers.DenseLayer(scalenet['input'], num_units=6, W=W, b=b, nonlinearity=None)
        scalenet['scale'] = TransformerLayer(scalenet['input'], scalenet['scale_init'], downsample_factor=1.0/7.0) # Output should be 3x224x224
        
        return scalenet, scalenet['scale']
        
    def build_vgg_model(self, input_var):
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

        print_info("Building VGG-16 model...")

        net = {}
        
        net['input'] = InputLayer(shape = (None, 3, 224, 224), input_var = input_var, name = 'vgg_input')
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

        print_info("Loading VGG16 pre-trained weights from file 'vgg16.pkl'...")
        with open('vgg16.pkl', 'rb') as f:
            params = pickle.load(f)

        #net_output.initialize_layers()
        lasagne.layers.set_all_param_values(net['prob'], params['param values'])
        print_info("Alright, pre-trained VGG16 model is ready!")

        return net, net['prob']

    def build_loss(self, input_var, target_var, deterministic=False):
        from lasagne.layers import get_output
        from lasagne.objectives import squared_error

        # Compute good ol' L2-norm loss between prediction and target
        network_output = get_output(self.network_out, deterministic=deterministic)
        l2_loss = squared_error(network_output, target_var).mean()

        # Compute loss from VGG's intermediate layers
        x_scaled = get_output(self.input_scaled_out, deterministic=deterministic)
        y_scaled = get_output(self.target_scaled_out, deterministic=deterministic)

        layers = [self.vgg_model['conv1_1'], self.vgg_model['conv2_1'], self.vgg_model['conv3_1'], self.vgg_model['conv4_2']]
        x_1, x_2, x_3, x_4 = get_output(layers, inputs=x_scaled, deterministic=deterministic)
        y_1, y_2, y_3, y_4 = get_output(layers, inputs=y_scaled, deterministic=deterministic)

        loss_conv1_1 = squared_error(x_1, y_1).mean()
        loss_conv2_1 = squared_error(x_2, y_3).mean()
        loss_conv3_1 = squared_error(x_3, y_3).mean()
        loss_conv4_2 = squared_error(x_4, y_4).mean()

        return l2_loss + 0.001*loss_conv1_1 + 0.001*loss_conv2_1 + 0.005*loss_conv3_1 + 0.01*loss_conv4_2
    
