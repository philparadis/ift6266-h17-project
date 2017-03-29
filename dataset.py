import os
import cPickle as pkl
import numpy as np
import PIL.Image as Image

### Define utility functions
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
        self.images_outer2d = normalize_data(self.images_outer2d)
        self.images_inner2d = normalize_data(self.images_inner2d)
        self._is_normalized = True

    def denormalize(self):
        if not self._is_normalized:
            print("WARNING: Attempting to denormalize already denormalized dataset... Ignoring this call...")
            return
        self.images_outer_flat = denormalize_data(self.images_outer_flat)
        self.images_inner_flat = denormalize_data(self.images_inner_flat)
        self.images_outer2d = denormalize_data(self.images_outer2d)
        self.images_inner2d = denormalize_data(self.images_inner2d)
        self._is_normalized = False
    
    def read_jpgs_and_captions_and_flatten(self, paths_list, caption_path, force_reload = False):
        # Get a list of all training images full filename paths
        if 
        print("Loading images paths from: " + settings.TRAIN_DIR + "/*.jpg")
        train_images_paths = glob.glob(settings.TRAIN_DIR + "/*.jpg")
        print("Found %i image paths." % len(train_images_paths))
        print("Loading images and captions data into memory and performing some pre-processing...")

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
                
                if i % 10000 == 0:
                    print("Loaded image #%i" % i)

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

    def load_flattened(test_size = 0.2, rand_seed=1):
        ### Split into training and testing data
        from sklearn.cross_validation import train_test_split
        num_rows = Dataset.images.shape[0]
        indices = np.arange(num_rows)
        id_train, id_test = train_test_split(indices,
                                             test_size=test_size,
                                             random_state=rand_seed)

        ### Generating the training and testing datasets (80%/20% train/test split)
        print("Splitting dataset into training and testing sets with shuffling...")
        X_train, X_test, Y_train, Y_test = Dataset.images_outer_flat[id_train], \
                                           Dataset.images_outer_flat[id_test], \
                                           Dataset.images_inner_flat[id_train], \
                                           Dataset.images_inner_flat[id_test]

        return X_train, X_test, Y_train, Y_test, id_train, id_test

    def load_2d(test_size = 0.2, rand_seed = 1):
        ### Split into training and testing data
        from sklearn.cross_validation import train_test_split
        num_rows = Dataset.images.shape[0]
        indices = np.arange(num_rows)
        id_train, id_test = train_test_split(indices,
                                             test_size=test_size,
                                             random_state=rand_seed)

        ### Generating the training and testing datasets (80%/20% train/test split)
        print("Splitting dataset into training and testing sets with shuffling...")
        X_train, X_test, Y_train, Y_test = Dataset.images_outer2d[id_train], \
                                           Dataset.images_outer2d[id_test], \
                                           Dataset.images_inner2d[id_train], \
                                           Dataset.images_inner2d[id_test]

        return X_train, X_test, Y_train, Y_test, id_train, id_test
