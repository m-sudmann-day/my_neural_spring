import numpy as np
import cv2
import os
import random
import math
from glob import glob
from misc import *
import cifar10
import mnist
import synth

from PIL import Image


class data_handler_source(Enum):
    PIPES = 1
    CIFAR10 = 2
    MNIST = 3
    SYNTH = 4

class batch_type(Enum):
    MINI = 1
    SAMPLE_WITH_REPLACEMENT=2
    SAMPLE_WITHOUT_REPLACEMENT=3

BATCH_AXIS_IMAGE_INDEX = 0
BATCH_AXIS_HEIGHT = 1
BATCH_AXIS_WIDTH = 2
BATCH_AXIS_CHANNEL = 3

class data_handler():

    def __init__(self, params):

        self.params = params
        self.train_ratio = self.params['train_ratio']
        self.Z_all = None
        self.Z_train = None
        self.Z_test = None

    def load_cfar10_images(self):
    
        (images, cls) = cifar10.load_data()

        #for x in range(0,60000,5000):
        #    # Prove that our images are reasonable.
        #    img = Image.fromarray(images[x], 'RGB')
        #    img = img.resize([128,128])
        #    img.show()

        return (images, cls, None)
    
    def load_mnist_images(self):
        
        (images, cls) = mnist.load_data()
        return (images, cls, None)
    
    def load_pipe_images(self, seed=None):
    
        images = list()
        labels = list()
        files = glob('../images/1*')
        files.extend(glob('../images/2*'))
        
        for file in files:
            
            filename = os.path.basename(file)
            label = int(filename[0])
            labels.append(label)
            
            im = cv2.imread(file)
            images.append(im)

        return (images, labels, files)
    
    def load_synth_images(self):

        return(synth.load_images())

    def merge_data(self, new_data):

        self.Z_all = new_data
        
    def maybe_shuffle_and_maybe_truncate_data(self, shuffle, shuffle_seed):
        
        nums = list(range(len(self.X_all)))

        if shuffle:
            if shuffle_seed is not None:
                random.seed(shuffle_seed)
            random.shuffle(nums)
        
        if self.params['max_images'] != None:
            nums = nums[:self.params['max_images']]
            
        self.X_all = [self.X_all[i] for i in nums]
        self.y_all = [self.y_all[i] for i in nums]
        if self.files_all != None:
            self.files_all = [self.files_all[i] for i in nums]
        
    def load_data(self, image_source, shuffle=True, shuffle_seed=None):
        
        if image_source == data_handler_source.PIPES:
            (self.X_all, self.y_all, self.files_all) = self.load_pipe_images()
        elif image_source == data_handler_source.CIFAR10:
            (self.X_all, self.y_all, self.files_all) = self.load_cfar10_images()
        elif image_source == data_handler_source.MNIST:
            (self.X_all, self.y_all, self.files_all) = self.load_mnist_images()
        elif image_source == data_handler_source.SYNTH:
            (self.X_all, self.y_all, self.files_all) = self.load_synth_images()
        
        self.image_source = image_source
        
        self.maybe_shuffle_and_maybe_truncate_data(shuffle, shuffle_seed)
        
        self.X_all = self.prepare_inputs(self.X_all)

        self.y_all = np.array(self.y_all).astype('int16')
        self.y_all -= min(self.y_all) # The code assumes the lowest label is a zero.
        #TODO mapping of categories to labels
        
        self.num_image_channels = self.X_all.shape[BATCH_AXIS_CHANNEL]
        self.num_classes = len(set(self.y_all))

        # Calculations for train and test sizes based on train_ratio.
        self.num_available_samples = self.y_all.shape[0]
        
        self.encode_outputs()
        
    def prepare_inputs(self, images):

        X = np.stack(images)

        # Convert to float32 to match the expectation of the model.
        X = X.astype('float32')

        # Pixel values range in [0, 255].  Translate into [-1, 1].
        # Instead of doing this on the entire set at once, it is done individually
        # on each image so that those images that do not use the full range of
        # input values get their values stretched to fill the range.  This could
        # help accommodate different lighting conditions.  For an RGB image, this
        # would have to be done across the three layers, not separately for each layer.
        
        #TODO: handle noncontiguous label values
        for i in range(X.shape[0]):
            one_image = X[i]
            max = one_image.max()
            min = one_image.min()
            #TODO: RuntimeWarning: invalid value encountered in true_divide (when using CIFAR subset)
            X[i] = 2 * ((one_image - min) / (max - min)) - 1
        return X

    def encode_outputs(self):
        
        self.y_all = to_categorical(self.y_all, self.num_classes)
        # Use one-hot encoding, i.e. "categorical".
        #self.y_train = to_categorical(self.y_train, self.num_classes)
        #self.y_test = to_categorical(self.y_test, self.num_classes)

    def split_one_array(self, arr):

        if arr is None:
            return (None, None)
        else:
            train = arr[:self.num_train_samples]
            test = arr[self.num_train_samples:(self.num_train_samples + self.num_test_samples)]
            return(train, test)
        
    def split_data(self):
    
        self.num_train_samples = math.floor(self.num_available_samples * self.train_ratio)
        self.num_test_samples = self.num_available_samples - self.num_train_samples

        (self.files_train, self.files_test) = self.split_one_array(self.files_all)
        self.files_all = None
        (self.X_train, self.X_test) = self.split_one_array(self.X_all)
        self.X_all = None
        (self.y_train, self.y_test) = self.split_one_array(self.y_all)
        self.y_all = None
        (self.Z_train, self.Z_test) = self.split_one_array(self.Z_all)
        self.Z_all = None

        # Reshape the data so that they are dimensioned like real images (layer/X/Y).
        # I might have x and y coordinates reversed, but we're dealing with square images anyway.
        shape = self.X_train.shape
        self.num_pixels_X = shape[BATCH_AXIS_WIDTH]
        self.num_pixels_Y = shape[BATCH_AXIS_HEIGHT]

    def get_stats(self):
        
        return({'num_available_samples':self.num_available_samples,
                'num_classes':self.num_classes,
                'num_image_channels':self.num_image_channels,
                'num_pixels_X':self.num_pixels_X,
                'num_pixels_Y':self.num_pixels_Y,
                'num_test_samples':self.num_test_samples,
                'num_train_samples':self.num_train_samples})

    def start_epoch(self, batch_type, batch_size, validation_set=False):
        
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.batch_num = 0
        self.batch_offset = 0
        self.batches_from_validation_set = validation_set
        
        if self.batches_from_validation_set:
            self.batch_num_samples = self.num_test_samples
        else:
            self.batch_num_samples = self.num_train_samples
            
    def is_epoch_finished(self):
        
        return (self.batch_offset >= self.batch_num_samples)
    
    def get_next_batch(self):
        
        self.batch_num += 1
        batch_end = self.batch_offset + self.batch_size
        
        if batch_end > self.batch_num_samples:
            batch_end = self.batch_num_samples
        
        batch_Z = None
        
        if self.batches_from_validation_set:
            batch_X = self.X_test[self.batch_offset:batch_end]
            batch_y = self.y_test[self.batch_offset:batch_end]
            if not self.Z_test is None:
                batch_Z = self.Z_test[self.batch_offset:batch_end]
        else:
            batch_X = self.X_train[self.batch_offset:batch_end]
            batch_y = self.y_train[self.batch_offset:batch_end]
            if not self.Z_train is None:
                batch_Z = self.Z_train[self.batch_offset:batch_end]

        prev_batch_offset = self.batch_offset
        self.batch_offset = batch_end

        return (prev_batch_offset, batch_X, batch_y, batch_Z)
    