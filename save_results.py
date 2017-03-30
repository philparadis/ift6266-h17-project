### Utilities functions

import os
import numpy as np
import PIL.Image as Image
from keras.utils import plot_model

import settings

## Your model will be saved in:                           models/<experiment_name>.h5
## A summary of your model architecture will saved be in: models/summary_<experiment_name>.txt
## Your model's performance will be saved in:             models/performance_<experiment_name>.txt

def save_model_info(model):
    out_dir = "models/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    model.save(os.path.join(out_dir, settings.EXP_NAME + '.h5')) 
    
    plot_model(model, to_file=os.path.join(out_dir, 'architecture_' + settings.EXP_NAME + '.png'), show_shapes=True)
    
    old_stdout = sys.stdout
    sys.stdout = open(os.path.join(out_dir, 'summary_' + settings.EXP_NAME + '.txt'), 'w')
    model.summary()
    sys.stdout = old_stdout

    with open(os.path.join(out_dir, 'performance_' + settings.EXP_NAME + '.txt'), 'w') as fd:
        # evaluate the model
        scores = model.evaluate(X_train, Y_train, batch_size=settings.BATCH_SIZE)
        fd.write("Training score %s: %.4f\n" % (model.metrics_names[1], scores[1]))
        scores = model.evaluate(X_test, Y_test, batch_size=settings.BATCH_SIZE)
        fd.write("Testing score %s: %.4f\n" % (model.metrics_names[1], scores[1]))
        
## Your predictions will be saved in: predictions/assets/<experiment_name>/Y_pred_<i>.jpg
##                                    predictions/assets/<experiment_name>/Y_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_outer_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_<i>.jpg
##                                    predictions/assets/<experiment_name>/X_full_pred_<i>.jpg
def save_predictions_info(pred, pred_indices, dataset,
                          num_images = 20, use_flattened_datasets = True):
    if use_flattened_datasets:
        out_dir = os.path.join('predictions/', "assets/", settings.EXP_NAME)
        if not os.path.exists(out_dir):
            print("Creating new directory to save predictions results: " + out_dir)
            os.makedirs(out_dir)
        else:
            print("Overwriting previously saved prediction results in directory: " + out_dir)
            
        for row in range(num_images):
            idt = pred_indices[row]
            Image.fromarray(dataset.images_outer2d[idt]).save(os.path.join(out_dir, 'images_outer2d_' + str(row) + '.jpg'))
            Image.fromarray(pred[row]).save(os.path.join(out_dir, 'images_pred_' + str(row) + '.jpg'))
            Image.fromarray(dataset.images_inner2d[idt]).save(os.path.join(out_dir, 'images_inner2d_' + str(row) + '.jpg'))
            Image.fromarray(dataset.images[idt]).save(os.path.join(out_dir, 'fullimages_' + str(row) + '.jpg'))
            fullimg_pred = np.copy(dataset.images[idt])
            center = (int(np.floor(fullimg_pred.shape[0] / 2.)), int(np.floor(fullimg_pred.shape[1] / 2.)))
            fullimg_pred[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = pred[row, :, :, :]
            Image.fromarray(fullimg_pred).save(os.path.join(out_dir, 'fullimages_pred_' + str(row) + '.jpg'))
    else:
        raise NotImplementedError("Haven't implemented save_predictions_info for 2D images (only flattened images).")

def print_results_as_html(pred, num_images=20):
    html_dir = os.path.join("predictions/")
    img_src = os.path.join("assets/", settings.EXP_NAME)
    path_html = os.path.join(html_dir, "results_" + settings.EXP_NAME + ".html")
    print("Saving results as html to: " + path_html)

    with open(path_html, 'w') as fd:
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
