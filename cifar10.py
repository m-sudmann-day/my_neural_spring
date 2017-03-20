########################################################################
#
# # LARGELY STRIPPED DOWN VERSION OF THE FOLLOWING:
#
########################################################################
#
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "../all_images/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def load_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    all_images = None
    all_classes = None

    # For each data-file.
    for file in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        
        path = os.path.join(data_path, "cifar-10-batches-py/" + file)
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        fo.close()

        image_batch = dict[b'data']
        # ? images of shape 3x32x32
        image_batch = image_batch.reshape([-1, 3, 32, 32])
        # transposed to ? images of 32x32x3
        image_batch = image_batch.transpose([0,2,3,1])

        if all_images is None:
            all_images = image_batch
        else:
            all_images = np.concatenate((all_images, image_batch))

        class_batch = dict[b'labels']

        if all_classes is None:
            all_classes = class_batch
        else:
            all_classes = np.concatenate((all_classes, class_batch))

    return all_images, all_classes

