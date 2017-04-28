import os, sys, errno
import json
import numpy as np
from termcolor import cprint

### Utility functions to manipule numpy datasets

def normalize_data(data):
    if data.dtype == 'float32':
        return
    data = data.astype('float32')
    data /= 255
    return data

def denormalize_data(data):
    if data.dtype == 'uint8':
        return
    data *= 255
    data = data.astype('uint8')
    return data

def unflatten_to_4tensor(data, num_rows, width = 64, height = 64, is_colors_channel_first = True):
    if is_colors_channel_first:
        return data.reshape(num_rows, 3, width, height)
    else:
        return data.reshape(num_rows, width, height)

def unflatten_to_3tensor(data, width = 64, height = 64, is_colors_channel_first = True):
    if is_colors_channel_first:
        return data.reshape(3, width, height)
    else:
        return data.reshape(width, height, 3)
    
def transpose_colors_channel(data, from_first_to_last = True):
    if from_first_to_last:
        ### Convert colors channel from first position to last position
        if len(data.shape) == 4:
            ### We are dealing with a batch (4-tensor)
            if data.shape[1] != 3 and data.shape[3] == 3:
                raise Exception("It appears that your colors channel is located last. Pass argument from_first_to_last=False.")
            num_rows = data.shape[0]
            width = data.shape[2]
            height = data.shape[3]
            return data.transpose(0, 2, 3, 1).reshape(num_rows, width, height, 3)
        elif len(data.shape) == 3:
            ### We are dealing with a single image (3-tensor)
            if data.shape[0] != 3 and data.shape[2] == 3:
                raise Exception("It appears that your colors channel is located last. Pass argument from_first_to_last=False.")
            width = data.shape[1]
            height = data.shape[2]
            return data.transpose(1, 2, 0).reshape(width, height, 3)
        else:
            raise ValueError("Dataset is not a 4-tensor batch or a 3-tensor image, as expected.")
    else:
        ### Convert colors channel from last position to first position
        if len(data.shape) == 4:
            ### We are dealing with a batch (4-tensor)
            if data.shape[3] != 3 and data.shape[1] == 3:
                raise Exception("It appears that your colors channel is located first. " +\
                                "You need to use 'transpose_to_colors_channel_last' instead.")
            num_rows = data.shape[0]
            width = data.shape[1]
            height = data.shape[2]
            return data.transpose(0, 3, 1, 2).reshape(num_rows, 3, width, height)
        elif len(data.shape) == 3:
            ### We are dealing with a single image (3-tensor)
            if data.shape[2] != 3 and data.shape[0] == 3:
                raise Exception("It appears that your colors channel is located first. " + \
                                "You need to use 'transpose_to_colors_channel_last' instead.")
            width = data.shape[0]
            height = data.shape[1]
            return data.transpose(2, 0, 1).reshape(3, width, height)
        else:
            raise ValueError("Dataset is not a 4-tensor batch or a 3-tensor image, as expected.")


### Pretty exceptions handling and pretty logging messages

def handle_exceptions(msg, e, type, fg = None, bg = None, attrs = []):
    from settings import VERBOSE, MODULE_HAVE_XTRACEBACK

    cprint("(!) {0}: {1}".format(type, msg), fg, bg, attrs=attrs)
    # Write the exception reason/message in Magenta
    sys.stdout.write("\033[35m(!) Reason: ")
    print(e)
    # Return to normal color
    sys.stdout.write("\033[30m")

    if not MODULE_HAVE_XTRACEBACK:
        raise e
    elif VERBOSE >= 1:
        # If verbose is not 0, print the trace
        # Print in Magenta
        from traceback import print_exc

        print("\033[35m(!) Traceback of exception:\033[30m")
        print_exc()

def handle_critical(msg, e):
    handle_exceptions(msg, e, type = "CRITICAL", fg = "white", bg = "on_red", attrs=["bold", "underline"])

def print_critical(msg):
    cprint("(!) {0}: {1}".format("WARNING", msg), "white", "on_red", attrs=["bold", "underline"])

def handle_error(msg, e):
    handle_exceptions(msg, e, type = "ERROR", fg = "red")

def print_error(msg):
    cprint("(!) {0}: {1}".format("ERROR", msg), "red")

def handle_warning(msg, e):
    handle_exceptions(msg, e, type = "WARNING", fg = "yellow")

def print_warning(msg):
    cprint("(!) {0}: {1}".format("WARNING", msg), "yellow")
                      
def print_info(msg):
    cprint("(!) {0}: {1}".format("INFO", msg), "cyan")

def print_positive(msg):
    cprint("(!) {0}: {1}".format("GOOD", msg), "cyan", attrs=["bold"])
                      
def force_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(dst)
            os.symlink(src, dst)

def get_json_pretty_print(json_object):
    return json.dumps(json_object, sort_keys=True, indent=4, separators=(',', ': '))

### Saving results in convenient formats

### Utilities functions

import os
from os.path import join
import sys
import numpy as np
import PIL.Image as Image
from keras.utils import plot_model

## Your model will be saved in:                           models/<experiment_name>.hdf5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

## Your predictions will be saved in: predictions/assets/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/assets/<experiment_name>/Y_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_pred_<i>.jpg
def save_keras_predictions(pred, pred_indices, dataset,
                           num_images = 20, use_flattened_datasets = True):
    from settings import touch_dir
    from settings import ASSETS_DIR

    if use_flattened_datasets:
        touch_dir(ASSETS_DIR)
        print_positive("Saving result images (outer frame input, inner frame prediction, true inner frame, and combination of outer frame + prediction and outer frame + true inner frame) within the directory: {}".format(ASSETS_DIR))
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
    from settings import touch_dir
    from settings import HTML_DIR

    touch_dir(HTML_DIR)
    img_src = "assets/"
    html_file = join(HTML_DIR, "results.html")
    print_positive("Saving results as html to: " + html_file)
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
