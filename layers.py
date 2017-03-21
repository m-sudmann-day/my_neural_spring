import numpy as np
import sys
import tensorflow as tf
import os
import cv2
import math
import data_handler
from max_activation_set import *
from my_array import *
from numpy import pad
from batch_norm import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    #return (grad * (op > 0.).astype(float32) * (grad > 0.).astype(float32))
    return(tf.where(grad < 0., tf.zeros(grad.get_shape()), gen_nn_ops._relu_grad(grad, op.outputs[0])))

@ops.RegisterGradient("LeakyRelu")
def leaky_relu(input_pt):
  return tf.where(tf.greater(input_pt, 0.0), input_pt, 0.01 * input_pt)
  
class layer():

    name = None
    extent = None
    num_output_channels = None
    content = None
    content_tf = None
    weights = None
    weights_shape = None
    biases = None
    input_layer = None

    def __init__(self, name, input_layer):
        
        self.name = name
        self.input_layer = input_layer
        
    def my_relu(self, tensor, guided_backprop, tensor_guided_backprop):

        if (guided_backprop):
            with tf.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
                return tf.nn.relu(tensor, name="guided_relu")
        else:
            return tf.nn.relu(tensor, name="normal_relu")
        
        def normal_relu(): return tf.nn.relu(tensor, name="normal_relu")
        
        with tf.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
            def guided_relu(): return tf.nn.relu(tensor, name="guided_relu")
        
        return(tf.cond(tensor_guided_backprop, guided_relu, normal_relu, name="relu"))

    def has_multiple_input_channels(self):

        if self.weights_shape is None: return False
        if not isinstance(self.weights_shape, tuple): return False
        if len(self.weights_shape) != 3: return False
        return True
    
    def get_weights_by_input_channel(self, weight_values):

        result = list()
        input_has_multiple_channels = self.input_layer is not None and self.input_layer.has_multiple_input_channels()
        
        for i in range(weight_values.shape[1]):

            weights_subset = weight_values[:,i]
            
            if input_has_multiple_channels:
                weights_subset = np.reshape(weights_subset, self.input_layer.weights_shape)
            else:
                weights_subset = np.reshape(weights_subset, (1, 1, self.input_layer.num_neurons))
            
            result.append(weights_subset)

        return(result)

class image_layer(layer):
    
    def __init__(self, name, input_tensor, trainable):
        super().__init__(name, None)

        with tf.name_scope(name):
            
            shape = input_tensor.get_shape().as_list()
            self.extent = shape[data_handler.BATCH_AXIS_HEIGHT]
            self.num_output_channels = shape[data_handler.BATCH_AXIS_CHANNEL]

            if trainable: # the content gets copied from the input tensor
                # TODO : get the number of channels from the input tensor
                varr = tf.Variable(tf.zeros([3, shape[1], shape[2], shape[3]]), name='image_var', trainable=True)
                #self.content_tf = tf.stop_gradient(varr, 'stop')
                self.assign_op = tf.assign(varr, input_tensor, name='assign_image_var')
                self.content_tf = varr
            else: # the content IS the input tensor
                self.content_tf = input_tensor
            
    def get_summary(self):
        
        return "IMAGE {0}: ({1},{2},{3})".format(self.name, self.extent, self.extent, self.num_output_channels)
        
    def get_activations_for_one_image(self, output_values, image_num):
        
        return(output_values[image_num])
    
    def get_tile(self, image_num_in_batch, x1, y1, x2, y2):
    
        x_dim = self.content.shape[1]
        y_dim = self.content.shape[2]
        z_dim = self.content.shape[3]

        # A lazy way to handle the padding needed as the tile may be requested from beyond the edges of the image.        
        temp = np.full((x_dim*3, y_dim*3, z_dim), self.content.arr.min(), dtype=np.float)
        temp[x_dim:x_dim*2, y_dim:y_dim*2, :] = self.content.arr[image_num_in_batch]
        return temp[x_dim+x1:x_dim+x2+1, y_dim+yself.layers['image'].content_tf1:y_dim+y2+1, :]
    
    def calculate_image_coordinates(self, x1, y1, x2, y2):

        return(x1, y1, x2, y2)

class merge_layer(layer):
    
    def __init__(self, name, input_layer, merge_tensor):
        super().__init__(name, input_layer)

        merge_shape = merge_tensor.get_shape().as_list()
        
        print ("***", merge_shape)
        print("***", input_layer.content_tf.get_shape())
        
        self.extent = input_layer.extent
        self.num_input_channels = input_layer.num_output_channels
        self.num_merge_channels = merge_shape[data_handler.BATCH_AXIS_CHANNEL]
        self.num_output_channels = self.num_input_channels + self.num_merge_channels
        
        self.content_tf = tf.concat(data_handler.BATCH_AXIS_CHANNEL, [self.input_layer.content_tf, merge_tensor])

    def get_summary(self):
        
        return "MERGE {0}: ({1},{2},{3}) + ({1},{2},{4}) -> ({1},{2},{5})".format(self.name, self.extent, self.extent,
                                                                                  self.num_input_channels, self.num_merge_channels, self.num_output_channels)

class conv_layer(layer):
    
    def __init__(self, name, input_layer, params, num_filters, filter_extent, trainable, guided_backprop, tensor_guided_backprop):
        super().__init__(name, input_layer)
            
        self.dropout_ratio = params.dropout_ratio_conv
        self.input_layer = input_layer
        self.num_input_channels = input_layer.num_output_channels
        self.num_output_channels = num_filters
        self.filter_extent = filter_extent
        self.extent = input_layer.extent
        self.weights_shape = (self.extent, self.extent, self. num_output_channels)

        with tf.name_scope(name):
            
            #self.weights = tf.Variable(tf.random_normal([filter_extent, filter_extent, self.num_input_channels, num_filters]), name="w", trainable=trainable)
            #self.biases = tf.Variable(tf.random_normal([num_filters]), name="b", trainable=trainable)
            num_neurons_in = filter_extent * filter_extent * input_layer.num_output_channels
            num_neurons_out = filter_extent * filter_extent * self.num_output_channels
            stddev = math.sqrt(2.0 / (num_neurons_in + num_neurons_out))
            initial_weights = tf.truncated_normal([filter_extent, filter_extent, self.num_input_channels, num_filters], stddev=stddev)
            initial_biases = tf.fill([num_filters], 0.1)

            self.weights = tf.Variable(initial_weights, name="w", trainable=trainable)
            self.biases = tf.Variable(initial_biases, name="b", trainable=trainable)


            self.conv = tf.nn.conv2d(input_layer.content_tf, self.weights, strides=[1, 1, 1, 1], padding='SAME')
            self.content_tf = self.conv + self.biases
            self.content_tf = batch_norm(self.content_tf, True, None)
            self.content_tf = self.my_relu(self.content_tf, guided_backprop, tensor_guided_backprop)
                        
            if self.dropout_ratio > 0.0:
                self.content_tf = tf.nn.dropout(self.content_tf, 1.0 - self.dropout_ratio)

        self.max_activations_per_output_channel = list()
        for i in range(self.num_output_channels):
            max_act_set = max_activation_set(self, 100)
            self.max_activations_per_output_channel.append(max_act_set)

    def handle_activations_for_batch(self, first_image_num):
        
        for oc_content, oc_max_activations in zip(self.content.slices('oc'), self.max_activations_per_output_channel):
            oc_max_activations.handle_activations(oc_content, first_image_num)

        for max_activation_set in self.max_activations_per_output_channel:
            max_activation_set.sort_and_truncate()
        
        for max_activation_set in self.max_activations_per_output_channel:
            max_activation_set.get_tiles(first_image_num)

    def get_summary(self):
        
        return "CONV {0}: ({1},{2},{3}) -> ({4},{5},{6})".format(self.name, self.input_layer.extent, self.input_layer.extent, self.num_input_channels, self.extent, self.extent, self.num_output_channels)
        
    def get_activations_for_one_image(self, output_values, image_num):
        
        return(output_values[image_num])

    def get_weights_by_input_channel(self, weight_values):

        result = list()
                
        for i in range(weight_values.shape[2]):
            result.append(weight_values[:,:,i,:])

        return(result)

    def get_tile(self, image_num_in_batch, x1, y1, x2, y2):

        #TODO handle stride other than 1
        pad = int((self.filter_extent - 1) / 2)
        return self.input_layer.get_tile(image_num_in_batch, x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    
    def calculate_image_coordinates(self, x1, y1, x2, y2):

        #TODO handle stride other than 1
        pad = int((self.filter_extent - 1) / 2)
        return self.input_layer.calculate_image_coordinates(x1 - pad, y1 - pad, x2 + pad, y2 + pad)

class maxpool_layer(layer):
    
    def __init__(self, name, input_layer, pool_size):
        super().__init__(name, input_layer)

        with tf.name_scope(name):
            
            if not (input_layer.extent / pool_size).is_integer():
                sys.stderr.write("WARNING Maxpool layer '{0}' has an input extent of {1} which is not a multiple of its pool size {2}.\n".format(name, input_layer.extent, pool_size))
                sys.stderr.write("The layer is not created.\n")
    
                self.content_tf = input_layer.content_tf
                self.extent = input_layer.extent

            
            else:
                
                self.content_tf = tf.nn.max_pool(input_layer.content_tf, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')
                self.extent = int(input_layer.extent / pool_size)
    
            self.pool_size = pool_size
            self.input_layer = input_layer
            self.num_output_channels = input_layer.num_output_channels
            self.weights_shape = (self.extent, self.extent, self. num_output_channels)
            
    def get_summary(self):
        
        return "MAXPOOL {0}: ({1},{2},{3}) -> ({4},{5},{6})".format(self.name, self.input_layer.extent, self.input_layer.extent, self.input_layer.num_output_channels, self.extent, self.extent, self.num_output_channels)
        
    def get_activations_for_one_image(self, output_values, image_num):
        
        return(output_values[image_num])

    def get_weights_by_input_channel(self, weight_values):

        return list()
    
    def get_tile(self, image_num_in_batch, x1, y1, x2, y2):

        return self.input_layer.get_tile(image_num_in_batch, x1 * self.pool_size, y1 * self.pool_size,
                                                            (x2 + 1) * self.pool_size - 1, (y2 + 1) * self.pool_size - 1)
    
    def calculate_image_coordinates(self, x1, y1, x2, y2):

        return self.input_layer.calculate_image_coordinates(x1 * self.pool_size, y1 * self.pool_size,
                                                            (x2 + 1) * self.pool_size - 1, (y2 + 1) * self.pool_size - 1)

class fc_layer(layer):

    def __init__(self, name, input_layer, params, num_neurons, trainable, guided_backprop, tensor_guided_backprop):
        super().__init__(name, input_layer)
            
        self.dropout_ratio = params.dropout_ratio_fc

        with tf.name_scope(name):
            
            #TODO: wrap following line in a conditional checking if the input_layer is maxpool or conv
            num_input_nodes = input_layer.extent * input_layer.extent * input_layer.num_output_channels
            
            #self.weights = tf.Variable(tf.random_normal([num_input_nodes, num_neurons]), name="w", trainable=trainable)
            #self.biases = tf.Variable(tf.random_normal([num_neurons]), name="b", trainable=trainable)
            stddev = math.sqrt(2.0 / (num_input_nodes + num_neurons))
            initial_weights = tf.truncated_normal([num_input_nodes, num_neurons], stddev=stddev)
            initial_biases = tf.fill([num_neurons], 0.1)
            self.weights = tf.Variable(initial_weights, name="w", trainable=trainable)
            self.biases = tf.Variable(initial_biases, name="b", trainable=trainable)

            self.content_tf = tf.reshape(input_layer.content_tf, [-1, num_input_nodes])
            self.content_tf = tf.matmul(self.content_tf, self.weights) + self.biases
            self.content_tf = batch_norm(self.content_tf, False, None)
            self.content_tf = self.my_relu(self.content_tf, guided_backprop, tensor_guided_backprop)
            
            if self.dropout_ratio > 0.0:
                self.content_tf = tf.nn.dropout(self.content_tf, 1.0 - self.dropout_ratio)
            
            self.num_neurons = num_neurons
            self.weights_shape = (self.num_neurons)

    def get_activations_for_one_image(self, output_values, image_num):
        
            #TODO : this only handles one image and that image must be num 0
            output_values = np.expand_dims(output_values, 0)
            return(output_values)
            
    def get_summary(self):
        return "FC {0}: ({1},{2},{3}) -> ({4})".format(self.name, self.input_layer.extent, self.input_layer.extent, self.input_layer.num_output_channels, self.num_neurons)

class output_layer(layer):
    
    def __init__(self, name, input_layer, num_neurons, trainable):
        super().__init__(name, input_layer)
        
        with tf.name_scope(name):
            
            self.weights = tf.Variable(tf.random_normal([input_layer.num_neurons, num_neurons]), name="w", trainable=trainable)
            self.biases = tf.Variable(tf.random_normal([num_neurons]), name="b", trainable=trainable)
            self.num_neurons = num_neurons
            self.content_tf = tf.matmul(input_layer.content_tf, self.weights) + self.biases

            if not trainable:
                #self.pretested_output_act = tf.placeholder(tf.float32, [None, num_neurons], name='pretested_output_act')
                self.pretested_output_overlay = tf.placeholder(tf.float32, [None, num_neurons], name='pretested_output_overlay')
                #self.pretested_output_act_overlaid = tf.multiply(self.pretested_output_act, self.pretested_output_overlay)
                self.content_tf_overlaid = tf.multiply(self.content_tf, self.pretested_output_overlay)
                self.content_tf_overlaid_sum = tf.reduce_sum(self.content_tf_overlaid)
                self.cost2 = tf.negative(tf.reduce_sum(self.content_tf_overlaid)) #tf.reduce_sum(tf.subtract(self.pretested_output_act_overlaid, self.content_tf_overlaid))

    def get_summary(self):
        
        return "OUTPUT {0}: ({1}) -> ({2})".format(self.name, self.input_layer.num_neurons, self.num_neurons)

    def get_activations_for_one_image(self, output_values, image_num):
        
            #TODO : this only handles one image and that image must be num 0
            output_values = np.expand_dims(output_values, 0)
            return(output_values)
