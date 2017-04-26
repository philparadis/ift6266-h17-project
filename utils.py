import os, sys, errno
import json
import traceback
import xtraceback
from termcolor import cprint

def handle_exceptions(msg, e, type, fg = None, bg = None, attrs = []):
    from settings import VERBOSE

    cprint("(!) {0: >8}: {1}".format(type, msg), fg, bg, attrs=attrs)
    # Write the exception reason/message in Magenta
    sys.stdout.write("\033[35m(!) Reason: ")
    print(e)
    # Return to normal color
    sys.stdout.write("\033[30m")
    # If verbose is not 0, print the trace
    if VERBOSE >= 1:
        # Print in Magenta
        print("\033[35m(!) Traceback of exception:\033[30m")
        traceback.print_exc()

def handle_critical(msg, e):
    handle_exceptions(msg, e, type = "CRITICAL", fg = "white", bg = "on_red", attrs=["bold", "underline"])

def print_critical_msg(msg):
    cprint("(!) {0: >9}: {1}".format("WARNING", msg), "white", "on_red", attrs=["bold", "underline"])

def handle_error(msg, e):
    handle_exceptions(msg, e, type = "ERROR", fg = "red")

def print_error(msg):
    cprint("(!) {0: >9}: {1}".format("ERROR", msg), "red")

def handle_warning(msg, e):
    handle_exceptions(msg, e, type = "WARNING", fg = "yellow")

def print_warning(msg):
    cprint("(!) {0: >9}: {1}".format("WARNING", msg), "yellow")
                      
def print_info(msg):
    cprint("(!) {0: >8}: {1}".format("INFO", msg), "cyan")

def print_positive(msg):
    cprint("(!) {0: >8}: {1}".format("GOOD", msg), "green")
                      
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

## Your model will be saved in:                           models/<experiment_name>.h5
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
