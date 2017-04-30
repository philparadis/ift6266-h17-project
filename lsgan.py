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
from utils import print_critical, print_error, print_warning, print_info, print_positive, log

class LSGAN_Model(GAN_BaseModel):
    def __init__(self, model_name, hyperparams = hyper_params.default_lsgan_hyper_params):
        super(LSGAN_Model, self).__init__(model_name = model_name, hyperparams = hyperparams)
        self.train_fn = None
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

        
    def initialize(self):
        pass
        

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

    def train(self, dataset, num_epochs = 1000, epochsize = 50, batchsize = 64, initial_eta = 0.0001, architecture = 7):
        """You can choose architecture = 1 through 7."""
        import lasagne
        import theano.tensor as T
        from theano import shared, function
        # Load the dataset
        log("Fetching data...")

        X_train, X_val, y_train, y_val, ind_train, ind_test = dataset.return_data()

        # Prepare Theano variables for inputs and targets
        noise_var = T.matrix('noise')
        input_var = T.tensor4('inputs')

        # Create neural network model
        log("Building model and compiling functions...")
        generator, gen_layers = self.build_generator(noise_var, architecture = architecture)
        critic, critic_layers = self.build_critic(input_var, architecture = architecture)

        if settings.FEATURE_MATCHING > 1:
            # Create expression for passing real data through the critic
            real_out = lasagne.layers.get_output(critic_layers[-settings.FEATURE_MATCHING])
            # Create expression for passing fake data through the critic
            fake_out = lasagne.layers.get_output(critic_layers[-settings.FEATURE_MATCHING],
                                                 lasagne.layers.get_output(generator))
        else:
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
        if self.optimizer == "rmsprop":
            generator_updates = lasagne.updates.rmsprop(generator_loss, generator_params, learning_rate=eta)
            critic_updates = lasagne.updates.rmsprop(critic_loss, critic_params, learning_rate=eta)
        else: #adam
            generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=eta, beta1=0.5)
            critic_updates = lasagne.updates.adam(critic_loss, critic_params, learning_rate=eta, beta1=0.5)

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

        # Create experiment's results directories
        settings.touch_dir(settings.BASE_DIR)
        settings.touch_dir(settings.EPOCHS_DIR)
        settings.touch_dir(settings.CHECKPOINTS_DIR)
        settings.touch_dir(settings.MODELS_DIR)

        # Set variables to control training
        epoch_eta_threshold = min(num_epochs // 10, 250)
        max_very_low_loss = 20
        num_very_low_loss = 0
        num_low_loss = 0
        ratio_gen_critic = 1

        # Finally, launch the training loop.
        print_positive("Starting training of LSGAN model!")
        # We create an infinite supply of batches (as an iterable generator):
        batches = self.iterate_minibatches(X_train, y_train, batchsize, shuffle=True, forever=True)

        # We iterate over epochs:
        found_stop_file = False
        generator_updates = 0
        next_epoch_checkpoint = settings.EPOCHS_PER_CHECKPOINT
        for epoch in range(num_epochs):
            start_time = time.time()

            print_info("Epoch {} out of {}:".format(epoch, num_epochs))

            num_critics_update = epochsize
            # In each epoch, we do `epochsize` generator and critic updates.
            if num_very_low_loss > 0 or num_low_loss > 0:
                rebalance_ratio = min(float(max_very_low_loss - num_very_low_loss) / float(max_very_low_loss),
                                      max(1.0-float(num_low_loss)/10.0, 0))
                num_critics_update = int(max(float(epochsize)*rebalance_ratio, 5))

            if num_low_loss > 0:
                print_warning("Nmber of successive low critics loss is now: {0}".format(num_low_loss))
            if num_very_low_loss > 0:
                print_critical("Number of successive *extremely low* critics loss is now: {0}.".format(num_very_low_loss))
            if num_critics_update != epochsize:
                print_warning("Rebalancing losses: {} critics updates VS {} generator updates.".format(num_critics_update, epochsize))
                                                       
            critic_losses = []
            generator_losses = []
            for u in range(epochsize):
                inputs, targets = next(batches)
                if u < num_critics_update:
                    critic_losses.append(critic_train_fn(inputs))
                generator_losses.append(generator_train_fn())

            ### Balance out gen and critic losses if necessary
            if ratio_gen_critic > 4:
                extra = int(math.ceil(ratio_gen_critic
                                      * (2*num_very_low_loss + 1)
                                      * (num_low_loss + 1)))
                extra = min(extra, epochsize)
                log("   Perfoming {} extra rounds of generator training to rebalance the losses.".format(extra))
                for _i in range(extra):
                    generator_losses.append(generator_train_fn())
            elif ratio_gen_critic < 0.25:
                extra = int(math.ceil(1/ratio_gen_critic))
                extra = min(extra, epochsize)
                log("   Perfoming {} extra rounds of critic training to rebalance the losses.".format(extra))
                for _i in range(extra):
                    inputs, targets = next(batches)
                    critic_losses.append(critic_train_fn(inputs))


            # Then we print the results for this epoch:
            time_delta = time.time() - start_time
            self.wall_time += time_delta
            mean_generator_loss = np.mean(generator_losses)
            mean_critic_loss = np.mean(critic_losses)
            log("   epoch took {2:.3f} seconds".format(epoch + 1, num_epochs, time_delta))
            if self.wall_time < 7200:
                log("   total wall time is {:.2f} minutes".format(self.wall_time / 60))
            else:
                log("   total wall time is {:.2f} hours".format(self.wall_time / 3600))
            log("   generator loss = {}".format(mean_generator_loss))
            log("   critic loss    = {}".format(mean_critic_loss))

            if mean_critic_loss < 0.03:
                print_critical("The critic loss is extremely low. If the loss is below 0.03 for {} epochs in a row, the training will be aborted".format(max_very_low_loss))
                print_info("Attempting to balance out generator and critic...")
                ratio_gen_critic = mean_generator_loss / mean_critic_loss
                num_very_low_loss += 1
            elif mean_critic_loss < 0.07:
                print_warning("Critic loss is getting dangerously low!")
                print_info("Attempting to balance out generator and critic...")
                ratio_gen_critic = mean_generator_loss / mean_critic_loss
                num_low_loss += 1
            else:
                num_very_low_loss = 0
                num_low_loss = 0
                
            if num_very_low_loss >= max_very_low_loss:
                print_error("Extremely low critic loss obtained {} epochs in a row. Aborting training now.".format(max_very_low_loss))
                self.create_stop_file()

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


            # Check for STOP file
            if self.check_stop_file():
                found_stop_file = True
                print_warning("Detected STOP file in the experiment's base directory, "
                              + "after completing fully epoch {}. ".format(epoch + 1)
                              + "Updating checkpoint, saving models and aborting.")
                break

            if epoch >= next_epoch_checkpoint:
                ### Checkpoint time!!! (save model and checkpoint file)
                print_positive("CHECKPOINT AT EPOCH {}. Updating 'checkpoint.json' file and saving model...".format(epoch + 1))
                self.generator = generator
                self.discriminator = critic
                self.epochs_completed = epoch

                # Update checkpoint, saving model to disk at the same time
                self.update_checkpoint(settings.KEEP_ALL_CHECKPOINTS)

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
                    fd.write("generator loss  = {}\n".format(mean_generator_loss))
                    fd.write("critic loss     = {}\n".format(mean_critic_loss))
                # Set next checkpoint epoch
                next_epoch_checkpoint = epoch + settings.EPOCHS_PER_CHECKPOINT

            
            # After half the epochs, we start decaying the learn rate towards zero
            if epoch >= epoch_eta_threshold:
                progress = float(epoch - epoch_eta_threshold) / float(num_epochs)
                eta.set_value(lasagne.utils.floatX(initial_eta*math.pow(1 - progress, 1.5)))
            #if epoch >= min(num_epochs // 10, 500):
            #    progress = float(epoch) / float(num_epochs)
            #    eta.set_value(lasagne.utils.floatX(initial_eta*2*(1 - progress)))


        ### We are done training here!
        ### Time for a checkpoint!
        ### Checkpoint time!!! (save model and checkpoint file)
        if found_stop_file:
            print_warning("TRAINING ABORTED EARLY DUE TO A STOP FILE BEING CREATED!")
            print_info("Performing checkpoint update at epoch {}.".format(epoch + 1))
        else:
            print_positive("TRAINING COMPLETED SUCCESSFULLY! We reached the desired number of epochs of {} without a hitch. High five!".format(num_epochs))
            print_info("Performing checkpoint update at epoch {}.".format(epoch + 1))

        self.generator = generator
        self.discriminator = critic
        self.epochs_completed = epoch

        # Update checkpoint, saving model to disk at the same time
        print_info("Updating 'checkpoint.json' and saving both model's weights to disk...")
        self.update_checkpoint(settings.KEEP_ALL_CHECKPOINTS)

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
            fd.write("generator loss  = {}\n".format(mean_generator_loss))
            fd.write("critic loss     = {}\n".format(mean_critic_loss))


        ### Save model to class variables
        self.train_fn = train_fn
        self.gen_fn = gen_fn
        self.generator_train_fn = generator_train_fn
        self.critic_train_fn = critic_train_fn

        return generator, critic, train_fn, gen_fn

