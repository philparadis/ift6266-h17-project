# -*- coding: utf-8 -*-
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import losses
from keras import backend as K

import dcgan_lasagne

import settings
from save_results import *

class ModelParameters:
    def __init__(self, model_name, input_dim, output_dim, loss_function, learning_rate):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.is_trained = False


def build_mlp(model_params):
    model = Sequential()
    model.add(Dense(units=1024, input_shape=(model_params.input_dim, )))
    model.add(Activation('relu'))
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dense(units=model_params.output_dim))
    model_params.is_trained = False
    return model

def train_keras(model, model_params, Dataset):
    skip_ahead = False
    X_train, X_test, Y_train, Y_test, id_train, id_test = Dataset.return_data()

    if settings.RELOAD_MODEL == True:
        model_path = os.path.join(settings.MODELS_DIR, 'trained_model.h5')
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
            adam_optimizer = optimizers.Adam(lr=model_params.learning_rate) # Default lr = 0.001
            model.compile(loss=model_params.loss_function, optimizer=adam_optimizer, metrics=[model_params.loss_function])

            # Fit the model
            print("Fitting model...")
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                      epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

            # evaluate the model
            print("Evaluating model...")
            scores = model.evaluate(X_train, Y_train, batch_size=BATCH_SIZE)
            print("Training score %s: %.4f" % (model.metrics_names[1], scores[1]))
            scores = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
            print("Testing score %s: %.4f" % (model.metrics_names[1], scores[1]))
            model_params.is_trained = True

            #%% Save model
            save_model_info(model)
        else:
            model_path = os.path.join(settings.MODELS_DIR, 'trained_model.h5')
            print("Model was already trained, instead loading: " + model_path)
            model = load_model(model_path)
        
    return model

def build_conv_mlp(model_params):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 64, 64)
    else:
        input_shape = (64, 64, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dense(units=model_params.output_dim))

    model_params.is_trained = False

