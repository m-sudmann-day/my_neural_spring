#import numpy as np
#import sys
import tensorflow as tf
#import os
#import cv2
import data_handler
from max_activation_set import *
#from my_array import *
from params import *
import math
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import gen_nn_ops
from batch_norm import *
from relu import *
from dropout import *

class layerx:

    name = None
    params = None
    input_layer = None
    input_tensor = None
    input_shape = None
    output_tensor = None
    output_shape = None
    dropout_ratio = None
    weights = None
    weights_shape = None
    biases = None
    num_units = None
    num_units_kept = None
    trainable = None
    
    def __init__(self, name, params, input_layer):

        self.name = name
        self.params = params
        self.input_layer = input_layer
        if input_layer is not None:
            self.input_tensor = input_layer.output_tensor
            self.input_shape = input_layer.output_shape

    def get_summary(self):
        
        return "{0} {1}: {2} -> {3}".format(type(self).__name__, self.name, self.input_shape, self.output_shape)

class image_input_layerx(layerx):

    num_input_channels = None
    num_output_channels = None
    extent = None

    def __init__(self, name, params, input_tensor):

        super(image_input_layerx, self).__init__(name, params, None)

        self.input_tensor = input_tensor
        shape = self.input_tensor.get_shape().as_list()
        self.extent = shape[data_handler.BATCH_AXIS_HEIGHT]
        self.num_input_channels = shape[data_handler.BATCH_AXIS_CHANNEL]
        self.num_output_channels = self.num_input_channels
        self.num_units = self.extent * self.extent * self.num_output_channels
        self.num_units_kept = self.num_units
        self.trainable = (params.execution_mode in [execution_mode.TRAINING])
        self.output_shape = (self.extent, self.extent, self.num_output_channels)

        with tf.name_scope(self.name):

            if self.trainable:
                # the input gets copied from the input tensor into a variable
                # TODO : get the number of channels from the input tensor
                v = tf.Variable(tf.zeros([3, shape[1], shape[2], shape[3]]),
                    name='image_var', trainable=self.trainable)
                self.assign_op = tf.assign(v, input_tensor, name='assign_image_var')
                self.output_tensor = v
            else:
                # pass-through
                self.output_tensor = input_tensor
            
class image_conv_layerx(layerx):
    
    def __init__(self, name, params, input_layer, num_filters, filter_extent):
        super().__init__(name, params, input_layer)
        
        dropout_obj = dropout(params.dropout_ratio_conv)
        self.dropout_ratio = dropout_obj.ratio
        self.batch_normalization_method = params.batch_normalization_method
        self.filter_extent = filter_extent
        self.num_input_channels = input_layer.num_output_channels
        self.num_output_channels = num_filters
        self.extent = input_layer.extent
        self.output_shape = (self.extent, self.extent, self.num_output_channels)
        self.num_units = self.extent * self.extent * self.num_output_channels
        self.num_units_kept = dropout_obj.units_kept(self.num_units)
        self.trainable = (params.execution_mode in [execution_mode.TRAINING])

        with tf.name_scope(name):
            
            stddev = math.sqrt(2.0 / (self.input_layer.num_units_kept + self.num_units_kept))
            initial_weights = tf.truncated_normal([filter_extent, filter_extent, self.num_input_channels, self.num_output_channels], stddev=stddev)
            initial_biases = tf.fill([self.num_output_channels], 0.1)

            self.weights = tf.Variable(initial_weights, name="w", trainable=self.trainable)
            self.biases = tf.Variable(initial_biases, name="b", trainable=self.trainable)
            strides = [1, 1, 1, 1]

            T = input_layer.output_tensor
            T = tf.nn.conv2d(T, self.weights, strides=strides, padding='SAME')
            T = T + self.biases
            T = batch_normx(T, params, [0, 1, 2])
            T = relu(T, params)
            T = dropout_obj.create(T)

            self.output_tensor = T

        self.max_activations_per_output_channel = list()
        for i in range(self.num_output_channels):
            max_act_set = max_activation_set(self, 100)
            self.max_activations_per_output_channel.append(max_act_set)

class image_pool_layerx(layerx):
    
    def __init__(self, name, params, input_layer, pool_size):

        super().__init__(name, params, input_layer)

        self.pool_size = pool_size

        with tf.name_scope(name):
            
            if not (self.input_layer.extent / self.pool_size).is_integer():
                sys.stderr.write("WARNING Maxpool layer '{0}' has an input extent of {1} which is not a multiple of its pool size {2}.\n".format(name, input_layer.extent, pool_size))
                sys.stderr.write("The layer is not created.\n")
                quit()
            
            else:
                
                ksize = strides = [1, pool_size, pool_size, 1]
                self.output_tensor = tf.nn.max_pool(self.input_tensor, ksize=ksize, strides=strides, padding='SAME')
                self.extent = int(self.input_layer.extent / self.pool_size)
                self.num_input_channels = input_layer.num_output_channels
                self.num_output_channels = input_layer.num_output_channels
                self.num_units = input_layer.num_units
                self.num_units_kept = input_layer.num_units_kept
                self.output_shape = (self.extent, self.extent, self.num_output_channels)

class fc_layerx(layerx):

    def __init__(self, name, params, input_layer, num_units):

        super().__init__(name, params, input_layer)
        
        dropout_obj = dropout(params.dropout_ratio_fc)
        self.dropout_ratio = dropout_obj.ratio
        self.batch_normalization_method = params.batch_normalization_method
        self.num_units = num_units
        self.num_units_kept = dropout_obj.units_kept(self.num_units)
        self.output_shape = (self.num_units)

        trainable = (params.execution_mode == execution_mode.TRAINING)

        with tf.name_scope(name):
            
            # TODO: should I really only use kept units?
            # TODO: should I really use truncated normal? probably not
            stddev = math.sqrt(2.0 / (input_layer.num_units_kept + self.num_units_kept))
            initial_weights = tf.truncated_normal([input_layer.num_units, self.num_units], stddev=stddev)
            initial_biases = tf.fill([self.num_units], 0.01)

            self.weights = tf.Variable(initial_weights, name="w", trainable=trainable)
            self.biases = tf.Variable(initial_biases, name="b", trainable=trainable)

            T = input_layer.output_tensor
            T = tf.reshape(T, [-1, self.input_layer.num_units])
            T = tf.matmul(T, self.weights)
            T = T + self.biases
            T = batch_normx(T, params)
            T = relu(T, params)
            T = dropout_obj.create(T)
            
            self.output_tensor = T
