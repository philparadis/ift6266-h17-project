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
from utils import normalize_data, denormalize_data
from utils import denormalize_and_save_jpg_results, create_html_results_page

from models import BaseModel

class LasagneModel(BaseModel):
    def __init__(self, hyperparams = hyper_params.default_lasagne_hyper_params): 
        super(LasagneModel, self).__init__(hyperparams = hyperparams)
        self.network = None
        self.network_out = None
        self.list_matching_layers = []
        self.list_matching_layers_weights = []
        self.path_stop_file = os.path.join(settings.BASE_DIR, "STOP")
        
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
            
    def check_stop_file(self):
        """Return True if a file with name STOP is found in the base directory, return False otherwise"""
        return os.path.isfile(self.path_stop_file)

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
        train_loss = self.build_loss(input_var, target_var)

        # Update expressions
        from theano import shared
        eta = shared(lasagne.utils.floatX(settings.LEARNING_RATE))
        params = lasagne.layers.get_all_params(self.network_out, trainable=True)
        updates = lasagne.updates.adam(train_loss, params, learning_rate=eta)

        # Train loss function
        train_fn = theano.function([input_var, target_var], train_loss, updates=updates)

        # Test/validation Loss expression (disable dropout and so on...)
        test_loss = self.build_loss(input_var, target_var, deterministic=True)

        # Validation loss function
        val_test_fn = theano.function([input_var, target_var], test_loss)

        # Predict function
        predict_fn = theano.function([input_var],
                                     lasagne.layers.get_output(self.network_out, deterministic=True))
        
        # Finally, launch the training loop.
        log("Starting training...")
        batch_size = settings.BATCH_SIZE
        best_val_loss = 1.0e30
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
            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)
            log("Epoch {} of {} took {:.3f}s".format(epoch + 1, settings.NUM_EPOCHS, time.time() - start_time))
            log(" - training loss:    {:.6f}".format(mean_train_loss))
            log(" - validation loss:  {:.6f}".format(mean_val_loss))

            create_checkpoint = False

            if check_stop_file():
                create_checkpoint = True
            
            if epochs >= 8 and mean_val_loss < best_val_loss:
                best_val_loss_epoch = epoch + 1
                best_val_loss = mean_val_loss
                create_checkpoint = True
                print_positive("New best val loss = {:.6f}!!! Creating model checkpoint!".format(best_val_loss))
            elif epoch % settings.EPOCHS_PER_CHECKPOINT == 0:
                create_checkpoint = True
                print_info("Time for model checkpoint (every {} epochs)...".format(settings.EPOCHS_PER_CHECKPOINT))

            if create_checkpoint:
                # Save checkpoint
                model_checkpoint_filename = "model_checkpoint-val_loss.{:.6f}-epoch.{:0>3}.npz".format(best_val_loss, epoch + 1)
                model_checkpoint_path = os.path.join(settings.CHECKPOINTS_DIR, model_checkpoint_filename)
                print_info("Saving model checkpoint: {}".format(model_checkpoint_path))
                # Save model
                self.save_model(model_checkpoint_path)

            # Save samples for this epoch
            if epoch % settings.EPOCHS_PER_SAMPLES == 0:
                num_samples = 100
                num_rows = 10
                num_cols = 10
                samples = self.create_samples(X_val, y_val, batch_size, num_samples, predict_fn)
                samples = denormalize_data(samples)
                samples_path = os.path.join(settings.EPOCHS_DIR, 'samples_epoch_{0:0>5}.png'.format(epoch + 1))
                print_info("Time for saving sample images (every {} epochs)... ".format(settings.EPOCHS_PER_SAMPLES)
                           + "Saving {} sample images predicted validation dataset input images here: {}"
                           .format(num_samples, samples_path))
                try:
                    import PIL.Image as Image
                    Image.fromarray(samples.reshape(num_rows, num_cols, 3, 32, 32)
                                    .transpose(0, 3, 1, 4, 2)
                                    .reshape(num_rows*32, num_cols*32, 3)).save(samples_path)
                except ImportError as e:
                    print_warning("Cannot import module 'PIL.Image', which is necessary for the Lasagne model to output its sample images. You should really install it!")

            
            if check_stop_file():
                print_critical("STOP file found. Ending training here! Still producing results...")
                break


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

        # Save model's performance
        path_model_score = os.path.join(settings.PERF_DIR, "score.txt")
        print_info("Saving performance to file '{}'".format(path_model_score))
        log("")
        log("Performance statistics")
        log("----------------------")
        log(" * Model = {}".format(settings.MODEL))
        log(" * Number of training epochs = {0}".format(settings.NUM_EPOCHS))
        log(" * Final training score (metric: {0: >6})    = {1:.5f}".format(metric, np.mean(train_losses)))
        log(" * Final validation score  (metric: {0: >6}) = {1:.5f}".format(metric, np.mean(val_losses)))
        log(" * Best validation score   (metric: {0: >6}) = {1:.5f}".format(metric, best_val_loss))
        log(" * Epoch for best validation score           = {}"..format(best_val_loss_epoch))
        log(" * Testing dataset  (metric: {0: >6})        = {1:.5f}".format(metric, np.mean(test_losses)))
        log("")
        with open(path_model_score, "w") as fd:
            fd.write("Performance statistics\n")
            fd.write("----------------------\n")
            fd.write("Model = {}\n".format(settings.MODEL))
            fd.write("Number of training epochs = {0}\n".format(settings.NUM_EPOCHS))
            fd.write("Final training score (metric: {0: >6})    = {1:.5f}\n".format(metric, np.mean(train_losses)))
            fd.write("Final validation score  (metric: {0: >6}) = {1:.5f}\n".format(metric, np.mean(val_losses)))
            fd.write("Best validation score   (metric: {0: >6}) = {1:.5f}\n".format(metric, best_val_loss))
            fd.write("Epoch for best validation score           = {}\n"..format(best_val_loss_epoch))
            fd.write("Testing dataset  (metric: {0: >6})        = {1:.5f}\n".format(metric, np.mean(test_losses)))

        # Save predictions and create HTML page to visualize them
        num_images = 100
        test_images_original = np.copy(normalize_data(dataset.test_images))
        denormalize_and_save_jpg_results(preds, X_test, y_test, test_images_original, num_images)
        create_html_results_page(num_images)

    def create_samples(self, X, y, batch_size, num_samples, predict_fn):
        # Print the test error
        shuffle_indices = np.arange(X.shape[0])
        np.random.shuffle(shuffle_indices)
        if num_samples > X.shape[0]:
            num_samples = X.shape[0]
        X = X[shuffle_indices][0:num_samples]
        y = y[shuffle_indices][0:num_samples]

        samples = np.zeros((num_samples, 3, 32, 32))
        num_iter = 0
        for batch in self.iterate_minibatches(X, y, batch_size, shuffle=False):
            inputs, targets = batch
            samples[num_iter*batch_size:(num_iter+1)*batch_size] = predict_fn(inputs)
            num_iter += 1
        return samples

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
        from lasagne.layers import ReshapeLayer
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

        # net = {}
        # if self.use_dropout == True:
        #     net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        #     net['dropout1'] = DropoutLayer(net['input'], p=0.1)
        #     net['conv1'] = ConvLayer(net['dropout1'], 512, 5, stride=2, pad='same') # 32x32
        #     net['dropout2'] = DropoutLayer(net['conv1'], p=0.5)
        #     net['conv2'] = ConvLayer(net['dropout2'], 256, 7, stride=1, pad='same') # 32x32
        #     net['dropout3'] = DropoutLayer(net['conv2'], p=0.5)
        #     net['deconv1'] = Deconv2DLayer(net['dropout3'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        #     net['dropout4'] = DropoutLayer(net['deconv1'], p=0.5)
        #     net['deconv2'] = Deconv2DLayer(net['dropout4'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        #     net['dropout5'] = DropoutLayer(net['deconv2'], p=0.5)
        #     net['deconv3'] = Deconv2DLayer(net['dropout5'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        #     net['dropout6'] = DropoutLayer(net['deconv2'], p=0.5)
        #     net['deconv4'] = Deconv2DLayer(net['dropout6'], 3, 7, stride=1, crop='same', output_size=32, nonlinearity=sigmoid)
        # else:
        #     net['input'] = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
        #     net['conv1'] = ConvLayer(net['input'], 512, 5, stride=2, pad='same') # 32x32
        #     net['conv2'] = ConvLayer(net['conv1'], 256, 7, stride=1, pad='same') # 32x32
        #     net['deconv1'] = Deconv2DLayer(net['conv2'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        #     net['deconv2'] = Deconv2DLayer(net['deconv1'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        #     net['deconv3'] = Deconv2DLayer(net['deconv2'], 256, 7, stride=1, crop='same', output_size=32) # 32x32
        #     net['deconv4'] = Deconv2DLayer(net['deconv3'], 3, 7, stride=1, crop='same', output_size=32, nonlinearity=sigmoid)

        net = {}
        if self.use_dropout == True:
            net = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
            net = ConvLayer(net, 64, 3, stride=1, pad='same') #64x64
            net = ConvLayer(net, 64, 3, stride=1, pad='same') #64x64
            net = PoolLayer(net, 2) # 32x32
            net = ConvLayer(net, 64, 5, stride=1, pad='same') #32x32
            net = PoolLayer(net, 2) # 16x16
            net = ConvLayer(net, 96, 5, stride=1, pad='same') #16x16
            net = ConvLayer(net, 96, 5, stride=1, pad='same') #16x16
            net = PoolLayer(net, 2) # 8x8
            net = ConvLayer(net, 128, 3, stride=1, pad='same') #8x8
            net = ConvLayer(net, 128, 3, stride=1, pad='same') #8x8
            net = DropoutLayer(net, p=0.5)
            net = DenseLayer(net, 512)
            net = DropoutLayer(net, p=0.5)
            net = DenseLayer(net, 1024)
            net = DropoutLayer(net, p=0.5)
            net = DenseLayer(net, 3*32*32)
            net = DropoutLayer(net, p=0.5)
            net = ReshapeLayer(net, ([0], 3, 32, 32))
            net = ConvLayer(net, 3, 3, stride=1, pad='same', nonlinearity=sigmoid)
        else:
            net = InputLayer((batch_size, 3, 64, 64), input_var=input_var)
            net = ConvLayer(net, 64, 3, stride=1, pad='same') #64x64
            net = ConvLayer(net, 64, 3, stride=1, pad='same') #64x64
            net = PoolLayer(net, 2) # 32x32
            net = ConvLayer(net, 64, 5, stride=1, pad='same') #32x32
            net = PoolLayer(net, 2) # 16x16
            net = ConvLayer(net, 96, 5, stride=1, pad='same') #16x16
            net = ConvLayer(net, 96, 5, stride=1, pad='same') #16x16
            net = PoolLayer(net, 2) # 8x8
            net = ConvLayer(net, 128, 3, stride=1, pad='same') #8x8
            net = ConvLayer(net, 128, 3, stride=1, pad='same') #8x8
            net = DenseLayer(net, 512)
            net = DenseLayer(net, 1024)
            net = DenseLayer(net, 3*32*32)
            net = ReshapeLayer(net, ([0], 3, 32, 32))
            net = ConvLayer(net, 3, 3, stride=1, pad='same', nonlinearity=sigmoid)
            
        #self.network, self.network_out = net, net['deconv4']
        self.network, self.network_out = {}, net

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

