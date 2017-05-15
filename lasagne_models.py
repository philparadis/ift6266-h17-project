import os, sys, errno
import json
import time
import abc, six
import glob
import numpy as np

import lasagne
import theano
import theano.tensor as T

import utils
import hyper_params
import settings
from utils import handle_critical, handle_error, handle_warning
from utils import print_critical, print_error, print_warning, print_info, print_positive, log, logout
from utils import force_symlink, get_json_pretty_print
from utils import denormalize_and_save_jpg_results, create_html_results_page

from models import BaseModel

class LasagneModel(BaseModel):
    def __init__(self, hyperparams = hyper_params.default_lasagne_hyper_params): 
        super(LasagneModel, self).__init__(hyperparams = hyperparams)
        self.network = None
        self.network_out = None
        self.list_matching_layers = []
        self.list_matching_layers_weights = []
        
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
            
    def build_network(self, input_var, target_var):
        raise NotImplemented("This is an abstract base class. Please implement this function.")

    def build_loss(self, input_var, target_var, deterministic=False):
        raise NotImplemented("This is an abstract base class. Please implement this function.")

    def train(self, dataset):
        log("Fetching data...")
        X_train, X_val, y_train, y_val, ind_train, ind_val = dataset.return_train_data()
        X_test, y_test = dataset.return_test_data()
        
        #Variance of the prediction can be maximized to obtain sharper images.
        #If this coefficient is set to "0", the loss is just the L2 loss.
        StdevCoef = 0

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.tensor4('targets')

        # Create neural network model
        log("Building model and compiling functions...")
        self.build_network(input_var, target_var)

        # Build loss function
        loss = self.build_loss(input_var, target_var)

        # Update expressions
        from theano import shared
        eta = shared(lasagne.utils.floatX(settings.LEARNING_RATE))
        params = lasagne.layers.get_all_params(self.network_out, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=eta)

        # Test/validation Loss expression (disable dropout and so on...)
        test_loss = self.build_loss(input_var, target_var, deterministic=True)

        # Train loss function
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Validation loss function
        val_test_fn = theano.function([input_var, target_var], test_loss)

        # Predict function
        predict_fn = theano.function([input_var], test_prediction)
        
        # Finally, launch the training loop.
        log("Starting training...")
        batch_size = settings.BATCH_SIZE
        for epoch in range(settings.NUM_EPOCHS):
            start_time = time.time()
            train_losses = []
            for batch in self.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets = batch
                train_losses.append(train_fn(inputs, targets))
                
            val_losses = []
            for batch in self.iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                inputs, targets = batch
                val_losses.append(val_test_fn(inputs, targets))

            # Print the results for this epoch
            log("Epoch {} of {} took {:.3f}s".format(epoch + 1, settings.NUM_EPOCHS, time.time() - start_time))
            log(" - training loss:    {:.6f}".format(np.mean(train_losses)))
            log(" - validation loss:  {:.6f}".format(np.mean(val_losses)))


        print_info("Training complete!")
        # Print the test error
        test_losses = []
        num_iter = 0
        preds = np.zeros((X_test.shape[0], 3, 32, 32))
        for batch in self.iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
            inputs, targets = batch
            preds[num_iter*batch_size:(num_iter+1)*batch_size] = predict_fn(inputs)
            test_losses.append(val_test_fn(inputs, targets))
            num_iter += 1
        log("Final results:")
        log(" - test loss:        {:.6f}".format(np.mean(test_losses)))

        # Save model
        self.save_model(os.path.join(settings.MODELS_DIR, settings.EXP_NAME + ".npz"))

        # Save predictions and create HTML page to visualize them
        num_images = 100
        denormalize_and_save_jpg_results(preds, X_test, y_test, dataset.test_images, num_images)
        create_html_results_page(num_images)

    def save_model(self, filename):
        np.savez(filename, *lasagne.layers.get_all_param_values(self.network_out))

def rescale(x):
    return x*0.5
        
class Lasagne_Conv_Deconv(LasagneModel):
    def __init__(self, use_dropout=False):
        super(Lasagne_Conv_Deconv, self).__init__()
        self.use_dropout = use_dropout

    def build(self):
        pass

    def build_network(self, input_var, target_var):
        from lasagne.layers import InputLayer
        from lasagne.layers import DenseLayer
        from lasagne.layers import NonlinearityLayer
        from lasagne.layers import DropoutLayer
        from lasagne.layers import Pool2DLayer as PoolLayer
        from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
        from lasagne.nonlinearities import sigmoid, tanh

        try:            
            from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
        except ImportError as e:
            from lasagne.layers import Conv2DLayer as ConvLayer
            print_warning("Cannot import 'lasagne.layers.dnn.Conv2DDNNLayer' as it requires GPU support and a functional cuDNN installation. Falling back on slower convolution function 'lasagne.layers.Conv2DLayer'.")

        batch_size = settings.BATCH_SIZE
        
        net = {}
        # net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        # net['conv1'] = ConvLayer(net['input'], 128, 5, stride=2, pad='same') # 32x32
        # net['conv2'] = ConvLayer(net['conv1'], 256, 7, stride=2, pad='same') # 16x16
        # net['deconv1'] = Deconv2DLayer(net['conv2'], 128, 7, stride=1, crop='same', output_size=8) # 16x16
        # net['deconv2'] = Deconv2DLayer(net['deconv1'], 256, 7, stride=2, crop='same', output_size=16) # 32x32
        # net['deconv3'] = Deconv2DLayer(net['deconv2'], 256, 9, stride=1, crop='same', output_size=32) # 32x32
        # net['deconv4'] = Deconv2DLayer(net['deconv3'], 3, 9, stride=1, crop='same', output_size=32, nonlinearity=sigmoid)
        
        # net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        # net['conv1'] = ConvLayer(net['input'], 64, 5, pad=0)
        # net['pool1'] = PoolLayer(net['conv1'], 2) # 32x32
        # net['conv2'] = ConvLayer(net['pool1'], 128, 3, pad=0)
        # net['pool2'] = PoolLayer(net['conv2'], 2) # 16x16
        # net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=0)
        # net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=0)
        # net['pool3'] = PoolLayer(net['conv3_2'], 2) #8x8
        # net['conv4'] = ConvLayer(net['pool3'], 2048, 5, pad=0)
        # net['conv4_drop'] = NonlinearityLayer(net['conv4'], nonlinearity=rescale)
        # net['conv5'] = ConvLayer(net['conv4_drop'], 2048, 1, pad=0)
        # net['conv5_drop'] = NonlinearityLayer(net['conv5'], nonlinearity=rescale)

        # net['conv6'] = ConvLayer(net['conv5_drop'], 64, 1, pad=0, nonlinearity=sigmoid)
        # net['conv7'] = ConvLayer(net['conv6'], 10, 1, pad=0, nonlinearity=softmax4d)

        net = {}
        if use_dropout == True:
            net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
            net['dropout1'] = DropoutLayer(net['input'], p=0.1)
            net['conv1'] = ConvLayer(net['dropout1'], 256, 5, stride=2, pad='same') # 32x32
            net['dropout2'] = DropoutLayer(net['conv1'], p=0.5)
            net['conv2'] = ConvLayer(net['dropout2'], 256, 7, stride=2, pad='same') # 16x16
            net['dropout3'] = DropoutLayer(net['conv2'], p=0.5)
            net['deconv1'] = Deconv2DLayer(net['dropout3'], 256, 7, stride=1, crop='same', output_size=8) # 16x16
            net['dropout4'] = DropoutLayer(net['deconv1'], p=0.5)
            net['deconv2'] = Deconv2DLayer(net['dropout4'], 256, 7, stride=2, crop='same', output_size=16) # 32x32
            net['dropout5'] = DropoutLayer(net['deconv2'], p=0.5)
            net['deconv3'] = Deconv2DLayer(net['dropout5'], 256, 9, stride=1, crop='same', output_size=32) # 32x32
            net['dropout6'] = DropoutLayer(net['deconv2'], p=0.5)
            net['deconv4'] = Deconv2DLayer(net['dropout6'], 3, 9, stride=1, crop='same', output_size=32, nonlinearity=sigmoid)
        else:
            net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
            net['conv1'] = ConvLayer(net['input'], 256, 5, stride=2, pad='same') # 32x32
            net['conv2'] = ConvLayer(net['conv1'], 256, 7, stride=2, pad='same') # 16x16
            net['deconv1'] = Deconv2DLayer(net['conv2'], 256, 7, stride=1, crop='same', output_size=8) # 16x16
            net['deconv2'] = Deconv2DLayer(net['deconv1'], 256, 7, stride=2, crop='same', output_size=16) # 32x32
            net['deconv3'] = Deconv2DLayer(net['deconv2'], 256, 9, stride=1, crop='same', output_size=32) # 32x32
            net['deconv4'] = Deconv2DLayer(net['deconv3'], 3, 9, stride=1, crop='same', output_size=32, nonlinearity=sigmoid)

        self.network, self.network_out = net, net['deconv4']

    def build_loss(self, input_var, target_var, deterministic=False):
        # Training Loss expression
        network_output = lasagne.layers.get_output(self.network_out, deterministic=deterministic)
        loss = lasagne.objectives.squared_error(network_output, target_var).mean()
        return loss
        
class GAN_BaseModel(BaseModel):
    def __init__(self, hyperparams = hyper_params.default_gan_basemodel_hyper_params):
        super(GAN_BaseModel, self).__init__(hyperparams = hyperparams)
        self.generator = None
        self.discriminator = None
        self.train_fn = None
        self.gen_fn = None

        # Feature matching layers
        self.feature_matching_layers = []
        
        # Constants
        self.gen_filename = "model_generator.npz"
        self.disc_filename = "model_discriminator.npz"
        self.full_gen_path = os.path.join(settings.MODELS_DIR, self.gen_filename)
        self.full_disc_path = os.path.join(settings.MODELS_DIR, self.disc_filename)

    def build(self):
        pass

    def load_model(self):
        """Return True if a valid model was found and correctly loaded. Return False if no model was loaded."""
        import numpy as np
        from lasagne.layers import set_all_param_values

        if os.path.isfile(self.full_gen_path) and os.path.isfile(self.full_disc_path):
            print_positive("Found latest '.npz' model's weights files saved to disk at paths:\n{}\n{}".format(self.full_gen_path, self.full_disc_path))
        else:
            print_info("Cannot resume from checkpoint. Could not find '.npz'  weights files, either {} or {}.".format(self.full_gen_path, self.full_disc_path))
            return False
            
        try:
            ### Load the generator model's weights
            print_info("Attempting to load generator model: {}".format(self.full_gen_path))
            with np.load(self.full_gen_path) as fp:
                param_values = [fp['arr_%d' % i] for i in range(len(fp.files))]
            set_all_param_values(self.generator, param_values)

            ### Load the discriminator model's weights
            print_info("Attempting to load generator model: {}".format(self.full_disc_path))
            with np.load(self.full_disc_path) as fp:
                param_values = [fp['arr_%d' % i] for i in range(len(fp.files))]
            set_all_param_values(self.discriminator, param_values)
        except Exception as e:
            handle_error("Failed to read or parse the '.npz' weights files, either {} or {}.".format(self.full_gen_path, self.full_disc_path), e)
            return False
        return True

        
    def save_model(self, keep_all_checkpoints=False):
        """Save model. If keep_all_checkpoints is set to True, then a copy of the entire model's weight and optimizer state is preserved for each checkpoint, along with the corresponding epoch in the file name. If set to False, then only the latest model is kept on disk, saving a lot of space, but potentially losing a good model due to overtraining."""
        import numpy as np
        from lasagne.layers import get_all_param_values
        # Save the gen and disc weights to disk
        if keep_all_checkpoints:
            epoch_gen_path = "model_generator_epoch{0:0>4}.npz".format(self.epochs_completed)
            epoch_disc_path = "model_discriminator_epoch{0:0>4}.npz".format(self.epochs_completed)
            chkpoint_gen_path = os.path.join(settings.CHECKPOINTS_DIR, epoch_gen_path)
            chkpoint_disc_path = os.path.join(settings.CHECKPOINTS_DIR, epoch_disc_path)
            np.savez(chkpoint_gen_path, *get_all_param_values(self.generator))
            np.savez(chkpoint_disc_path, *get_all_param_values(self.discriminator))
            force_symlink("../checkpoints/{}".format(epoch_gen_path), self.full_gen_path)
            force_symlink("../checkpoints/{}".format(epoch_disc_path), self.full_disc_path)
        else:
            np.savez(self.full_gen_path, *get_all_param_values(self.generator))
            np.savez(self.full_disc_path, *get_all_param_values(self.discriminator))

