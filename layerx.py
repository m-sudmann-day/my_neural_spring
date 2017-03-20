import numpy as np
import sys
import tensorflow as tf
import os
import cv2
import data_handler
from max_activation_set import *
from my_array import *
from numpy import pad
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

class layerx:

    name = None
    input_layer = None
    input = None
    output = None

    def __init__(self, name, input_layer):

        self.name = name
        self.input_layer = input_layer
        if input_layer is not None:
            self.input = input_layer.output

class image_layerx(layerx):
    
    def __init__(self, name, image, trainable):

        super(image_layerx, self).__init__(None, None)
        self.input = image

        with tf.name_scope(name):
            
            shape = image.get_shape().as_list()
            self.extent = shape[data_handler.BATCH_AXIS_HEIGHT]
            self.num_output_channels = shape[data_handler.BATCH_AXIS_CHANNEL]

            if trainable: # the content gets copied from the image
                var = tf.Variable(tf.zeros([1, shape[1], shape[2], shape[3]]), name='image_var', trainable=True)
                self.assign_op = tf.assign(varr, image, name='assign_image_var')
                self.output = var
            else: # the content IS the image
                self.output = image
            
    def get_summary(self):
        
        return "IMAGE {0}: ({1},{2},{3})".format(self.name, self.extent, self.extent, self.num_output_channels)
        
    def get_activations_for_one_image(self, output_values, image_num):
        
        return(output_values[image_num])
    
    def get_tile(self, image_num_in_batch, x1, y1, x2, y2):
    
        x_dim = self.output.shape[1]
        y_dim = self.output.shape[2]
        z_dim = self.output.shape[3]

        # A lazy way to handle the padding needed as the tile may be requested from beyond the edges of the image.        
        temp = np.full((x_dim*3, y_dim*3, z_dim), self.output.arr.min(), dtype=np.float)
        temp[x_dim:x_dim*2, y_dim:y_dim*2, :] = self.output.arr[image_num_in_batch]
        return temp[x_dim+x1:x_dim+x2+1, y_dim+y1:y_dim+y2+1, :]
    
    def calculate_image_coordinates(self, x1, y1, x2, y2):

        return(x1, y1, x2, y2)
