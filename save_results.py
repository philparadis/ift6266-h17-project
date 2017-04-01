### Utilities functions

import os
from os.path import join
import sys
import numpy as np
import PIL.Image as Image
from keras.utils import plot_model

import settings
from settings import BASE_DIR
from settings import MODELS_DIR
from settings import EPOCHS_DIR
from settings import PERF_DIR
from settings import SAMPLES_DIR
from settings import PRED_DIR
from settings import ASSETS_DIR
from settings import HTML_DIR

## Your model will be saved in:                           models/<experiment_name>.h5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

def save_keras_model_info(model):
    settings.touch_dir(MODELS_DIR)
    ### Save model architecture and weights to .h5 file
    model.save(join(MODELS_DIR, 'trained_model.h5')) 
    ### Write an image that represents the model's architecture
    plot_model(model, to_file=join(MODELS_DIR, 'plot_model.png'), show_shapes=True)
    ### Output a summary of the model, including the various layers, activations and total number of weights
    old_stdout = sys.stdout
    sys.stdout = open(join(MODELS_DIR, 'summary.txt'), 'w')
    model.summary()
    sys.stdout = old_stdout


def save_keras_performance_results(model, model_params, X_train, Y_train, X_test, Y_test):
    ### Output training and testing scores
    settings.touch_dir(PERF_DIR)
    with open(join(PERF_DIR, 'losses.txt'), 'w') as fd:
        # evaluate the model
        fd.write("Loss function = %s\n" % model_params.loss_function)
        scores = model.evaluate(X_train, Y_train, batch_size=settings.BATCH_SIZE)
        fd.write("Training loss = %s: %.4f\n" % (model.metrics_names[1], scores[1]))
        scores = model.evaluate(X_test, Y_test, batch_size=settings.BATCH_SIZE)
        fd.write("Testing loss  = %s: %.4f\n" % (model.metrics_names[1], scores[1]))


## Your predictions will be saved in: predictions/assets/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/assets/<experiment_name>/Y_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_pred_<i>.jpg
def save_keras_predictions(pred, pred_indices, dataset,
                           num_images = 20, use_flattened_datasets = True):
    if use_flattened_datasets:
        settings.touch_dir(ASSETS_DIR)
        for row in range(num_images):
            idt = pred_indices[row]
            Image.fromarray(dataset.images_outer2d[idt]).save(join(ASSETS_DIR, 'images_outer2d_' + str(row) + '.jpg'))
            Image.fromarray(pred[row]).save(join(ASSETS_DIR, 'images_pred_' + str(row) + '.jpg'))
            Image.fromarray(dataset.images_inner2d[idt]).save(join(ASSETS_DIR, 'images_inner2d_' + str(row) + '.jpg'))
            Image.fromarray(dataset.images[idt]).save(join(ASSETS_DIR, 'fullimages_' + str(row) + '.jpg'))
            fullimg_pred = np.copy(dataset.images[idt])
            center = (int(np.floor(fullimg_pred.shape[0] / 2.)), int(np.floor(fullimg_pred.shape[1] / 2.)))
            fullimg_pred[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = pred[row, :, :, :]
            Image.fromarray(fullimg_pred).save(join(ASSETS_DIR, 'fullimages_pred_' + str(row) + '.jpg'))
    else:
        raise NotImplementedError("Haven't implemented save_predictions_info for 2D images (only flattened images).")

def print_results_as_html(pred, num_images=20):
    settings.touch_dir(HTML_DIR)
    img_src = "assets/"
    html_file = join(HTML_DIR, "results.html")
    print("Saving results as html to: " + html_file)
    with open(html_file, 'w') as fd:
        fd.write("""
<table>
  <tr>
    <th style="width:132px">Input</th>
    <th style="width:68px">Model prediction</th>
    <th style="width:68px">Correct output</th> 
    <th style="width:132px">Input + prediction</th>
    <th style="width:132px">Input + correct output</th>
  </tr>
""")

        for row in range(num_images):
            fd.write("  <tr>\n")
            fd.write("    <td><img src='%s/images_outer2d_%i.jpg' width='128' height='128'></td>\n" % (img_src, row))
            fd.write("    <td><img src='%s/images_pred_%i.jpg' width='64' height='64'></td>\n" % (img_src, row))
            fd.write("    <td><img src='%s/images_inner2d_%i.jpg' width='64' height='64'></td>\n" % (img_src, row))
            fd.write("    <td><img src='%s/fullimages_pred_%i.jpg' width='128' height='128'></td>\n" % (img_src, row))
            fd.write("    <td><img src='%s/fullimages_%i.jpg' width='128' height='128'></td>\n" % (img_src, row))
            fd.write('</tr>\n')
        
        fd.write('</table>')
