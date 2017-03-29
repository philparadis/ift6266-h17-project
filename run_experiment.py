import argparse
import os, sys
import glob
import numpy as np
import PIL.Image as Image
#from skimage.transform import resize

#################################################
# Run experiments here
# Define your global options and experiment name
# Then run the desired model
#################################################

### The experiment name is very important.

## Your model will be saved in:                           models/<experiment_name>.h5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

## Your predictions will be saved in: predictions/assets/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/assets/<experiment_name>/Y_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_pred_<i>.jpg

def run():
    model = None
    loss_function = None
    
    # Define model's specific settings
    if SETTINGS.model == "mlp":
        input_dim = 64*64*3 - 32*32*3
        output_dim = 32*32*3
        loss_function = "mse"
        model = models.build_mlp(input_dim, output_dim)
    else:
        raise NotImplementedError()
    
    experiment_name = "exp_model-%s_loss-%s_epochs-%i" \
                      % (model_name, loss_function, num_epochs)
    
    print("Experiment name = %s" % experiment_name)

    #######################################
    # Info about the dataset
    #######################################
    # The data is already split into training and validation datasets
    # The training dataset has:
    # - 82782 items
    # - 984 MB of data
    # The validation dataset has:
    # - 40504 items
    # - 481 MB of data
    #
    # There is also a pickled dictionary that maps image filenames (minutes the
    # .jpg extension) to a list of 5 strings (the 5 human-generated captions).
    # This dictionary is an OrderedDict with 123286 entries.

    ### Create and initialize an empty InpaintingDataset object
    Dataset = InpaintingDataset(input_dim, output_dim)

    ### Load dataset
    Dataset.read_jpgs_and_captions_and_flatten(train_images_paths, settings.CAPTIONS_PKL_PATH)

    print("Finished loading and pre-processing datasets...")
    print("Summary of datasets:")
    print("images.shape            = " + str(Dataset.images.shape))
    print("images_outer2d.shape    = " + str(Dataset.images_outer2d.shape))
    print("images_inner2d.shape    = " + str(Dataset.images_inner2d.shape))
    print("images_outer_flat.shape = " + str(Dataset.images_outer_flat.shape))
    print("images_inner_flat.shape = " + str(Dataset.images_inner_flat.shape))
    print("captions_ids.shape      = " + str(Dataset.captions_ids.shape))
    print("captions_dict.shape     = " + str(Dataset.captions_dict.shape))

     ###
    if SETTINGS.model == "mlp":
        model = train_mlp(model, Dataset)
    elif SETTINGS.model == "dcgan":
        model = train_dcgan(model, Dataset)
    
    ### Produce predictions
    Y_test_pred = model.predict(X_test, batch_size=batch_size)

    # Reshape predictions to a 2d image and denormalize data
    Y_test_pred = denormalize_data(Y_test_pred)
    num_rows = Y_test_pred.shape[0]
    Y_test_pred_2d = np.reshape(Y_test_pred, (num_rows, 32, 32, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type="string",
                        help="Model choice (current options: mlp, convnet, convnet_lstm, vae, dcgan)")
    parser.add_argument("experiment_name", type="string",
                        help="Name of experiment. Your results will be stored in subfolders with this name.")
    parser.add_argument("-v", "--verbose", action="store_true", type="int",
                        default=settings.VERBOSE, help="0 means quiet, 1 means verbose and 2 means limited verbosity.")
    parser.add_argument("-e", "--num_epochs", action="store", type="int",
                        default=settings.NUM_EPOCHS, help="Number of epochs to train")
    parser.add_argument("-b", "--batch_size", action="store", type="int",
                        default=settings.BATCH_SIZE, help="Size of minibatches")
    parser.add_argument("-l", "--load_model", action="store", type="string",
                        default=None, help="Load HF5 model from subdirectory 'models'. This will skip the training phase.")

    args = parser.parse_args()
    settings.MODEL = args.model
    settings.EXPERIMENT_NAME = args.experiment_name
    settings.NUM_EPOCHS = args.num_epochs
    settings.BATCH_SIZE = args.batch_size
    settings.VERBOSE = args.verbose

    if not args in ["mlp", "convnet", "convnet_lstm", "vae", "dcgan"]:
        raise NotImplementedError()
    
    run_experiment.run()
