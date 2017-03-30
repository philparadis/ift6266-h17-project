# -*- coding: utf-8 -*-
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import losses

import dcgan_lasagne

import settings
from save_results import *

class ModelParameters:
    def __init__(self, model_name, input_dim, output_dim, loss_function):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_function = loss_function
        self.is_trained = False


def build_mlp(model_params, input_dim, output_dim):
    model = Sequential()
    model.add(Dense(units=512, input_shape=(model_params.input_dim, )))
    model.add(Activation('relu'))
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dense(units=model_params.output_dim))
    model_params.is_trained = False
    return model

def train_mlp(model, model_params, Dataset, adam_lr=0.0005):
    skip_ahead = False
    
    ### Normalize datasets
    Dataset.normalize()
    
    X_train, X_test, Y_train, Y_test, id_train, id_test = Dataset.load_flattened()

    if settings.RELOAD_MODEL == True:
        model_path = os.path.join('models/', settings.EXP_NAME + '.h5')
        if os.path.isfile(model_path):
            print("The --load_model_from_file flag was passed and we found a matching model under '%s'." % model_path)
            print("Loading model from disk...")
            model = load_model(model_path)
            model_params.is_trained = True
            skip_ahead = True
        else:
            print("The --load_model_from_file flag was passed, but we cannot find any file matching '%s'." % model_path)

    if skip_ahead == False:
        if not model_params.is_trained:
            # Print model summary
            print("Model summary:")
            print(model.summary())

            # Compile model
            print("Compiling model...")
            adam_optimizer = optimizers.Adam(lr=adam_lr) # Default lr = 0.001
            model.compile(loss=model_params.loss_function, optimizer=adam_optimizer, metrics=[model_params.loss_function])

            # Fit the model
            print("Fitting model...")
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                      epochs=settings.NUM_EPOCHS, batch_size=settings.BATCH_SIZE, verbose=settings.VERBOSE)

            # evaluate the model
            print("Evaluating model...")
            scores = model.evaluate(X_train, Y_train, batch_size=settings.BATCH_SIZE)
            print("Training score %s: %.4f" % (model.metrics_names[1], scores[1]))
            scores = model.evaluate(X_test, Y_test, batch_size=settings.BATCH_SIZE)
            print("Testing score %s: %.4f" % (model.metrics_names[1], scores[1]))
            model_params.is_trained = True

            #%% Save model
            save_model_info(model)
        else:
            model_path = os.path.join('models/', settings.EXP_NAME + '.h5')
            print("Model was already trained, instead loading: " + model_path)
            model = load_model(model_path)
        
    ### Denormalize all datasets
    Dataset.denormalize()
    
    return model

def train_dcgan(Dataset):
    dcgan_lasagne.train(Dataset, settings.NUM_EPOCHS)
