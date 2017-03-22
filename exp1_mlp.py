# coding: utf-8
#!/usr/bin/env python2

"""
Created on Wed Mar 15 14:54:01 2017

@author: Philippe Paradis
"""

import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
#from skimage.transform import resize

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras import losses
from keras.utils import plot_model

import argparse

#################################################
# FIXED GLOBAL VARIABLES AND STATE VARIABLES
#################################################

### FIXED GLOBAL VARIABLES
input_dim = 64*64*3 - 32*32*3
output_dim = 32*32*3
path_mscoco="datasets/mscoco_inpainting/inpainting/"
path_traindata="train2014"
path_caption_dict="dict_key_imgID_value_caps_train_and_valid.pkl"


### STATE VARIABLES
is_dataset_loaded = False
is_model_trained = False


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


### Utilities functions

## Your model will be saved in:                           models/<experiment_name>.h5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt
def save_model_info(exp_name, model):
    out_dir = "models/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    model.save(os.path.join(out_dir, exp_name + '.h5')) 
    
    #TODO: INSTALL pydot
    #plot_model(model, to_file=os.path.join('model/', 'architecture_' + exp_name + '.png'), show_shapes=True)
    
    old_stdout = sys.stdout
    sys.stdout = open(os.path.join(out_dir, 'summary_' + exp_name + '.txt'), 'w')
    model.summary()
    sys.stdout = old_stdout

    with open(os.path.join(out_dir, 'performance_' + exp_name + '.txt'), 'w') as fd:
        # evaluate the model
        scores = model.evaluate(X_train, Y_train, batch_size=batch_size)
        fd.write("Training score %s: %.4f\n" % (model.metrics_names[1], scores[1]))
        scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
        fd.write("Testing score %s: %.4f\n" % (model.metrics_names[1], scores[1]))
        model.
        
## Your predictions will be saved in: predictions/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/<experiment_name>/Y_<i>.jpg
##                                    predictions/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/<experiment_name>/X_full_<i>.jpg
##                                    predictions/<experiment_name>/X_full_pred_<i>.jpg
def save_predictions_info(exp_name, pred, pred_indices, dataset,
                          num_images = 10, show_images = False, use_flattened_datasets = True):
    if use_flattened_datasets:
        out_dir = os.path.join('predictions/', exp_name, "assets/")
        if not os.path.exists(out_dir):
            print("Creating new directory to save predictions results: " + out_dir)
            os.makedirs(out_dir)
        else:
            print("Overwriting previously saved prediction results in directory: " + out_dir)
            
        for row in range(num_images):
            idt = pred_indices[row]
            Image.fromarray(dataset.images_outer2d[idt]).save(os.path.join(out_dir, 'images_outer2d_' + str(row) + '.jpg'))
            #img.show()

            Image.fromarray(pred[row]).save(os.path.join(out_dir, 'images_pred_' + str(row) + '.bmp'))
            #img.show()

            Image.fromarray(dataset.images_inner2d[idt]).save(os.path.join(out_dir, 'images_inner2d_' + str(row) + '.jpg'))
            #img.show()

            Image.fromarray(dataset.images[idt]).save(os.path.join(out_dir, 'fullimages_' + str(row) + '.jpg'))
            #fullimg.show()

            fullimg_pred = np.copy(dataset.images[idt])
            center = (int(np.floor(fullimg_pred.shape[0] / 2.)), int(np.floor(fullimg_pred.shape[1] / 2.)))
            fullimg_pred[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = pred[row, :, :, :]
            Image.fromarray(fullimg_pred).save(os.path.join(out_dir, 'fullimages_pred_' + str(row) + '.jpg'))
            #img.show()


def print_results_as_html(exp_name, pred, dataset, num_images=10):
    img_dir = os.path.join('assets/', exp_name)
    out_dir = os.path.join("predictions/", img_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    path_html = os.path.join("predictions/", "results_" + exp_name + ".html")
    print("Saving results as html to: " + path_html)

    with open(path_html, 'w') as fd:
        fd.write("""
<table style="width:150px">
  <tr>
    <th>Input (outer frame)</th>
    <th>Model prediction (inner frame)</th>
    <th>Correct output (inner frame)</th> 
    <th>Input + prediction</th>
    <th>Input + correct output)</th>
  </tr>
  <span class='icons'>
""")

        for row in range(num_images):
            fd.write("  <tr>\n")
            fd.write('    <td><img src="' + os.path.join(img_dir, '/images_outer2d_' + str(row) + '.jpg') + '" width="128" height="128"></td>\n')
            fd.write('    <td><img src="' + os.path.join(img_dir, '/images_pred_' + str(row) + '.jpg') + '" width="64" height="64"></td>\n')
            fd.write('    <td><img src="' + os.path.join(img_dir, '/images_inner2d_' + str(row) + '.jpg') + '" width="64" height="64"></td>\n')
            fd.write('    <td><img src="' + os.path.join(img_dir, '/fullimages_pred_' + str(row) + '.jpg') + '" width="128" height="128"></td>\n')
            fd.write('    <td><img src="' + os.path.join(img_dir, '/fullimages_' + str(row) + '.jpg') + '" width="128" height="128"></td>\n')
            fd.write('</tr>\n')

        fd.write('</table>')
        fd.write("<span class='icons'>")

def normalize_data(data):
    data = data.astype('float32')
    data /= 255
    return data

def denormalize_data(data):
    data *= 255
    data = data.astype('uint8')
    return data
    

### Define the main class for handling our dataset called InpaintingDataset

class InpaintingDataset(object):
    
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.images = []
        self.images_outer2d = []
        self.images_inner2d = []
        self.images_outer_flat = []
        self.images_inner_flat = []
        self.captions_ids = []
        self.captions_dict = []
        self._is_dataset_loaded = False
        self._is_flattened = False
        self._is_normalized = False
        self._num_rows = None
    
    def normalize(self):
        if self._is_normalized:
            print("WARNING: Attempting to normalize already normalized dataset... Ignoring this call...")
            return
        self.images_outer_flat = normalize_data(self.images_outer_flat)
        self.images_inner_flat = normalize_data(self.images_inner_flat)
        self._is_normalized = True

    def denormalize(self):
        if not self._is_normalized:
            print("WARNING: Attempting to denormalize already denormalized dataset... Ignoring this call...")
            return
        self.images_outer_flat = denormalize_data(self.images_outer_flat)
        self.images_inner_flat = denormalize_data(self.images_inner_flat)
        self._is_normalized = False
    
    def load_jpgs_and_captions_and_flatten(self, paths_list, caption_path, force_reload = False):
        with open(caption_path) as fd:
            caption_dict = pkl.load(fd)
        if not self._is_dataset_loaded and not force_reload:
            images = []
            images_outer2d = []
            images_inner2d = []
            images_outer_flat = []
            images_inner_flat = []
            captions_ids = []
            captions_dict = []
            for i, img_path in enumerate(paths_list):
                img = Image.open(img_path)
                img_array = np.array(img)

                # File names look like this: COCO_train2014_000000520978.jpg
                cap_id = os.path.basename(img_path)[:-4]

                ### Get input/target from the images
                center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
                if len(img_array.shape) == 3:
                    image = np.copy(img_array)

                    outer_2d = np.copy(img_array)
                    outer_2d[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0

                    outer = np.copy(img_array)
                    outer_mask = np.array(np.ones(np.shape(img_array)), dtype='bool')
                    outer_mask[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = False
                    outer_flat = outer.flatten()
                    outer_mask_flat = outer_mask.flatten()
                    outer_flat = outer_flat[outer_mask_flat]

                    inner2d = np.copy(img_array)
                    inner2d = inner2d[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]

                    inner = np.copy(img_array)
                    inner = inner[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
                    inner_flat = inner.flatten()
                else:
                    # For now, ignore greyscale images
                    continue
                    #X_outer = np.copy(img_array)
                    #X_outer[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
                    #X_inner = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]


                #Image.fromarray(img_array).show()
                images.append(image)
                images_outer2d.append(outer_2d)
                images_inner2d.append(inner2d)
                images_outer_flat.append(outer_flat)
                images_inner_flat.append(inner_flat)
                captions_ids.append(cap_id)
                captions_dict.append(caption_dict[cap_id])

            self.images = np.array(images)
            self.images_inner_flat = np.array(images_inner_flat)
            self.images_outer_flat = np.array(images_outer_flat)
            self.images_outer2d = np.array(images_outer2d)
            self.images_inner2d = np.array(images_inner2d)
            self.captions_ids = np.array(captions_ids)
            self.captions_dict = np.array(captions_dict)

            self._is_flattened = True
            self._is_dataset_loaded = True
            self._num_rows = self.images.shape[0]
        else:
            print("Dataset is already loaded. Skipping this call. Please pass the argument force_reload=True to force reloading of dataset.")


### Create and initialize an empty InpaintingDataset object
Dataset = InpaintingDataset(input_dim, output_dim)


### Load training images and captions

# Get captions dictionary path
caption_path = os.path.join(mscoco, dict_key_captions)
    
# Get a list of all training images full filename paths
data_path = os.path.join(path_mscoco, path_traindata)
print("Loading images from: " + data_path + "/*.jpg")
train_images_paths = glob.glob(data_path + "/*.jpg")
Dataset.load_jpgs_and_captions_and_flatten(train_images_paths, caption_path)

print("Finished loading and pre-processing datasets...")
print("Summary of datasets:")
print("images.shape            = " + str(Dataset.images.shape))
print("images_outer2d.shape    = " + str(Dataset.images_outer2d.shape))
print("images_inner2d.shape    = " + str(Dataset.images_inner2d.shape))
print("images_outer_flat.shape = " + str(Dataset.images_outer_flat.shape))
print("images_inner_flat.shape = " + str(Dataset.images_inner_flat.shape))
print("captions_ids.shape      = " + str(Dataset.captions_ids.shape))
print("captions_dict.shape     = " + str(Dataset.captions_dict.shape))


### Sanity check:
print("Performing sanity check using first 10 elements of first 3 rows:")
sanity_check_values = np.array([[57,   69,  57,  65,  79,  56,  63,  81,  43,  53],
                                [197, 202, 195, 167, 164, 147, 104,  87,  57, 102],
                                [104, 100,  97,  77,  80,  53, 172, 181, 128, 242]])
for i in range(3):
    top10 = Dataset.images_inner_flat[i, range(10)]
    print(top10)
    np.testing.assert_array_equal(top10, sanity_check_values[i])
    print("Row " + str(i) + " passed sanity check!")


### Normalize datasets
Dataset.normalize()

### Split into training and testing data
from sklearn.cross_validation import train_test_split
num_rows = Dataset.images.shape[0]
indices = np.arange(num_rows)
id_train, id_test = train_test_split(indices,
                                     test_size=0.20,
                                     random_state=1)

### Generating the training and testing datasets (80%/20% train/test split)
print("Splitting dataset into training and testing sets with shuffling...")
X_train, X_test, Y_train, Y_test = Dataset.images_outer_flat[id_train],                                    Dataset.images_outer_flat[id_test],                                    Dataset.images_inner_flat[id_train],                                    Dataset.images_inner_flat[id_test]

print("Splitting dataset into training and testing sets with shuffling...")
print("X_train.shape = " + str(X_train.shape))
print("X_test.shape  = " + str(X_test.shape))
print("Y_train.shape = " + str(Y_train.shape))
print("Y_test.shape  = " + str(Y_test.shape))
print("id_train.shape = " + str(id_train.shape))
print("id_test.shape  = " + str(id_test.shape))


### Sanity check:
print("id_train = " + str(id_train))
print("id_test  = " + str(id_test))
print("Y_train.shape = " + str(Y_train.shape))
print("Y_train[0,1500:1550] = \n" + str(Y_train[0,1500:1550]))

idx = id_train[0]
img = Image.fromarray(Dataset.images[0])
img.show()


if not is_model_trained:
    print("Creating MLP model...")
    # Create model
    model = Sequential()
    model.add(Dense(units=512, input_shape=(input_dim, )))
    model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(units=512))
    if use_sigmoid_final_layer:
        model.add(Activation('sigmoid'))
    else:
        model.add(Activation('relu'))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(units=output_dim))

    # Print model summary
    print("Model summary:")
    print(model.summary())

    # Compile model
    print("Compiling model...")
    adam_optimizer = optimizers.Adam(lr=0.0005) # Default lr = 0.001
    model.compile(loss=loss_function, optimizer=adam_optimizer, metrics=[loss_function])

    # Fit the model
    print("Fitting model...")
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs, batch_size=batch_size, verbose=2)

    # evaluate the model
    print("Evaluating model...")
    scores = model.evaluate(X_train, Y_train, batch_size=batch_size)
    print("Training score %s: %.2f" % (model.metrics_names[1], scores[1]))
    scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("Testing score %s: %.2f" % (model.metrics_names[1], scores[1]))
    is_model_trained = True

    #%% Save model
    save_model_info(experiment_name, model)
else:
    model_path = os.path.join('models/', experiment_name + '.h5')
    print("Model was already trained, instead loading: " + model_path)
    model = load_model(model_path)

### Produce predictions
Y_test_pred = model.predict(X_test, batch_size=batch_size)

# Reshape predictions to a 2d image and denormalize data
Y_test_pred = denormalize_data(Y_test_pred)
num_rows = Y_test_pred.shape[0]
Y_test_pred_2d = np.reshape(Y_test_pred, (num_rows, 32, 32, 3))

# Denormalize all datasets
Dataset.denormalize()

### Save predictions to disk
save_predictions_info(experiment_name, Y_test_pred_2d, id_test, Dataset)
print_results_as_html(experiment_name, Y_test_pred_2d, Dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParse(description="This is my IFT6266 project python architecture")

    ### The experiment name is very important.

    ## Your model will be saved in:                           models/<experiment_name>.h5
    ## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
    ## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

    ## Your predictions will be saved in: predictions/assets/<experiment_name>/images_pred_<i>.jpg
    ##                                    predictions/assets/<experiment_name>/images_inner2d_<i>.jpg
    ##                                    predictions/assets/<experiment_name>/images_outer2d_<i>.jpg
    ##                                    predictions/assets/<experiment_name>/fullimg_pred__<i>.jpg
    ##                                    predictions/assets/<experiment_name>/fullimg_<i>.jpg

    parser.add_argument('experiment_name', action="store", type=string, dest="option.experiment_name"))
    parser.add_argument('num_epochs', action="store", type=int, dest="options.num_epochs")

    parser.add_argument('--use_dropout', action="store_true", type=bool,
                        dest="option.use_dropout", default=False)
    parser.add_argument('--loss_function', action="store", type=string
                        dest="options.loss_function", default="mse")
    parser.add_argument("--output_num_images", action="store", type=int, 
                        dest="options.output_num_images", default=50)
    parser.add_argument("--sigmoid-final-layer", action="store_true", type=bool, 
                        dest="options.sigmoid_final_layer", default=False)
    parser.add_argument("--batch_sizes", action="store", type=int,
                        dest="options.batch_sizes", default=128)

    options = parser.parse_args()

    run_experiment(options)
