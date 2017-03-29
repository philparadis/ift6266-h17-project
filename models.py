from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.utils import plot_model
import dcgan_lasagne

is_model_trained = False

def build_mlp(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(units=512, input_shape=(input_dim, )))
    model.add(Activation('relu'))
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dense(units=output_dim))
    return model

def train_mlp(model, Dataset):
       ### Normalize datasets
    Dataset.normalize()
    
    X_train, X_test, Y_train, Y_test, id_train, id_test = Dataset.load_flattened()

    if not is_model_trained:
        # Print model summary
        print("Model summary:")
        print(model.summary())

        # Compile model
        print("Compiling model...")
        adam_optimizer = optimizers.Adam(lr=0.0005) # Default lr = 0.001
        model.compile(loss=loss_function, optimizer=adam_optimizer, metrics=[loss_function])

        # Fit the model
        print("Fitting model...")
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs, batch_size=batch_size, verbose=settings.VERBOSE)

        # evaluate the model
        print("Evaluating model...")
        scores = model.evaluate(X_train, Y_train, batch_size=batch_size)
        print("Training score %s: %.4f" % (model.metrics_names[1], scores[1]))
        scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
        print("Testing score %s: %.4f" % (model.metrics_names[1], scores[1]))
        is_model_trained = True

        #%% Save model
        save_model_info(experiment_name, model)
    else:
        model_path = os.path.join('models/', experiment_name + '.h5')
        print("Model was already trained, instead loading: " + model_path)
        model = load_model(model_path)
        
    # Denormalize all datasets
    Dataset.denormalize()
    
    return model

def train_dcgan(model, num_epochs, Dataset):
    dcgan_lasagne.train(num_epochs, 2e-4, Dataset)
