import sys
import platform
import numpy as np
import math
import tensorflow as tf
import os
from glob import glob
import random
import shutil
import matplotlib; matplotlib.use('Agg')
import pylab as pl
import time
import layers as lyr
import layerx as lyrx
from misc import *
from data_handler import *
from scipy.misc import imsave
from PIL import Image
from visualization import *
from visualization2 import *
from visualization3 import *
from my_array import *
from params import *

class model_tf():

    def __init__(self, params, history=None):

        self.g = tf.Graph()
        self.tf_vars = None
        self.params = params
        self.history = history        
        self.layers = dict()
        self.layers_as_list = list()
        self.layersx = dict()
        self.layersx_as_list = list()
        self.batch_type = batch_type.MINI
        self.using_restored_weights = False
        self.last_layerx = None

        #TODO: this does not result in reproducible behavior
        #         if 'seed' in self.params:
        #             seed = self.params['seed']
        #             random.seed(seed)
        #             np.random.seed(seed + 1)
        #             tf.set_random_seed(seed + 2)
            
    def add_layer(self, layer):

        self.layers[layer.name] = layer
        self.layers_as_list.append(layer)
        self.last_layer = layer
        
        return layer
            
    def add_layerx(self, layer):

        self.layersx[layer.name] = layer
        self.layersx_as_list.append(layer)
        self.last_layerx = layer
        
        return layer
    
    def create_model(self, dh, guided_backprop):
        
        assert self.g is tf.get_default_graph()
        self.trainable = not guided_backprop
        
        self.num_remaining_epochs = self.params.num_epochs
        self.num_completed_epochs = 0
        self.num_classes = dh.num_classes
        self.learning_rate = self.params.learning_rate
        self.batch_size = self.params.batch_size

        extent = dh.num_pixels_X

        tf.placeholder(tf.float32, [None, extent, extent, dh.num_image_channels], name='X')
        tf.placeholder(tf.float32, [None, self.num_classes], name='y')

        input = self.add_layer(lyr.image_layer("image", self.get_tensor_X(), not self.trainable))
        #self.add_layerx(lyrx.image_input_layerx("imagex", self.params, self.get_tensor_X()))

        if dh.data_source == data_source.MNIST:

            with tf.name_scope("convset1"):
                conv1a = self.add_layer(lyr.conv_layer("conv1a", self.last_layer, self.params, 16, 7, self.trainable, guided_backprop))
                #self.add_layerx(lyrx.image_conv_layerx("conv1a", self.params, self.last_layerx, 16, 7))
                conv1b = self.add_layer(lyr.conv_layer("conv1b", self.last_layer, self.params, 16, 7, self.trainable, guided_backprop))
                #self.add_layerx(lyrx.image_conv_layerx("conv1b", self.params, self.last_layerx, 16, 7))
            
            with tf.name_scope("convset2"):
                conv2a = self.add_layer(lyr.conv_layer("conv2a", self.last_layer, self.params, 16, 5, self.trainable, guided_backprop))
                #self.add_layerx(lyrx.image_conv_layerx("conv2a", self.params, self.last_layerx, 16, 5))
                conv2b = self.add_layer(lyr.conv_layer("conv2b", self.last_layer, self.params, 16, 3, self.trainable, guided_backprop))
                #self.add_layerx(lyrx.image_conv_layerx("conv2b", self.params, self.last_layerx, 16, 3))
                pool2 = self.add_layer(lyr.maxpool_layer("pool2", self.last_layer, 2))
                #self.add_layerx(lyrx.image_pool_layerx("pool2", self.params, self.last_layerx, 2))
                   
            with tf.name_scope("convset3"):
                conv3a = self.add_layer(lyr.conv_layer("conv3a", self.last_layer, self.params, 16, 3, self.trainable, guided_backprop))
                #self.add_layerx(lyrx.image_conv_layerx("conv3a", self.params, self.last_layerx, 16, 3))
                conv3b = self.add_layer(lyr.conv_layer("conv3b", self.last_layer, self.params, 16, 3, self.trainable, guided_backprop))
                #self.add_layerx(lyrx.image_conv_layerx("conv3b", self.params, self.last_layerx, 16, 3))

            fc1 = self.add_layer(lyr.fc_layer("fc1", self.last_layer, self.params, 128, self.trainable, guided_backprop))
            #self.add_layerx(lyrx.fc_layerx("fc1", self.params, self.last_layerx, 128))

        elif dh.data_source == data_source.CIFAR10:

            with tf.name_scope("convset1"):
                conv1a = self.add_layer(lyr.conv_layer("conv1a", self.last_layer, self.params, 64, 7, self.trainable, guided_backprop))
                conv1b = self.add_layer(lyr.conv_layer("conv1b", self.last_layer, self.params, 64, 7, self.trainable, guided_backprop))
                pool1 = self.add_layer(lyr.maxpool_layer("pool1", self.last_layer, 2))
            
            with tf.name_scope("convset2"):
                conv2a = self.add_layer(lyr.conv_layer("conv2a", self.last_layer, self.params, 96, 5, self.trainable, guided_backprop))
                conv2b = self.add_layer(lyr.conv_layer("conv2b", self.last_layer, self.params, 96, 3, self.trainable, guided_backprop))
                pool2 = self.add_layer(lyr.maxpool_layer("pool2", self.last_layer, 2))
                   
            with tf.name_scope("convset3"):
                conv3a = self.add_layer(lyr.conv_layer("conv3a", self.last_layer, self.params, 64, 3, self.trainable, guided_backprop))
                conv3b = self.add_layer(lyr.conv_layer("conv3b", self.last_layer, self.params, 64, 3, self.trainable, guided_backprop))
                pool3 = self.add_layer(lyr.maxpool_layer("pool3", self.last_layer, 2))

    # 
    #         if not dh.Z_train is None:
    #             num_merge_channels = dh.Z_train.shape[BATCH_AXIS_CHANNEL]
    #             tf.placeholder(tf.float32, [None, self.last_layer.extent, self.last_layer.extent, num_merge_channels], name='Z')
    # 
    #             input2 = self.add_layer(lyr.merge_layer("merge", self.last_layer, self.get_tensor_Z()))
    #             
    #             conv7 = self.add_layer(lyr.conv_layer("conv7", self.last_layer, 128, 5))
    #             conv8 = self.add_layer(lyr.conv_layer("conv8", self.last_layer, 128, 5))
    
                fc1 = self.add_layer(lyr.fc_layer("fc1", self.last_layer, self.params, 256, self.trainable, guided_backprop))
    
        output = self.add_layer(lyr.output_layer("output", self.last_layer, dh.num_classes, self.trainable))
        
        print("----MODEL SUMMARY----")
        for layer in self.layers_as_list:
            print(layer.get_summary())
        print("-------------------------")
        for layerx in self.layersx_as_list:
            print(layerx.get_summary())
        
        # Define loss and optimizer
        with tf.name_scope('softmax_'):
            self.probabilities = tf.nn.softmax_cross_entropy_with_logits(logits=output.content_tf, labels=self.get_tensor_y(), name='softmax')
            
        with tf.name_scope('cost_'):
            self.cost = tf.reduce_mean(self.probabilities, name='cost')
            #rightwrong = tf.add(-1., tf.multiply(self.get_tensor_y(), 2.))
            #self.cost3 = tf.reduce_sum(tf.multiply(rightwrong, output.content_tf), name='cost3')
            #self.cost2 = self.cost3 #tf.add(self.cost3, tf.nn.l2_loss(input.content_tf), name='cost2')

        # Evaluate model
        with tf.name_scope('evaluation'):
            pred = tf.argmax(output.content_tf, 1, name='pred')
            truth = tf.argmax(self.get_tensor_y(), 1, name='truth')
            correct_pred = tf.equal(pred, truth, name='is_correct')
            
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    def get_tensor_accuracy(self):
        return (self.g.get_tensor_by_name('evaluation/accuracy:0'))
    
    def get_tensor_cost(self):
        return (self.g.get_tensor_by_name('cost_/cost:0'))
    
    def get_tensor_X(self):
        return self.g.get_tensor_by_name('X:0')

    def get_tensor_y(self):
        return self.g.get_tensor_by_name('y:0')

    def get_tensor_Z(self):
        if 'Z:0' in self.g.as_graph_def().node:
            return self.g.get_tensor_by_name('Z:0')
        else:
            return None

    def get_savable_variables(self):
        
        vars = list()
        for layer in self.layers_as_list:
            if layer.weights is not None:
                vars.append(layer.weights)
            if layer.biases is not None:
                vars.append(layer.biases)
        return(vars)

    def load_or_init_variables(self, sess, dh):
        
        assert self.g is tf.get_default_graph()

        path = '../pretrained/' + str(dh.data_source).split('.')[1].lower()
        if os.path.exists(path):
            saver = tf.train.Saver(self.get_savable_variables())
            # An error here could be due to a mismatching output layer because a small dataset will not necessarily contain all labels.
            saver.restore(sess, os.path.join(path, 'vars.ckpt'))
            self.using_restored_weights = True
        else:
            sess.run(tf.global_variables_initializer())

        tf.assert_variables_initialized()

    def train_normal(self, dh):
        assert self.g is tf.get_default_graph()
        # Launch the graph
        #with tf.device('/gpu:0'):
        cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

        with tf.Session() as sess:
    
            summary_writer = tf.summary.FileWriter('../tboard/', self.g)

            log = { 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[] }

            #tf.summary.image('w_img', self.layers['conv1'].weights_image)
            summary_node = tf.summary.merge_all()
            summary_output = None

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
            objective = optimizer.minimize(self.get_tensor_cost())

            self.load_or_init_variables(sess, dh)

            self.tf_vars = self.get_variable_values(sess)
                
        while self.num_remaining_epochs >= 0:

            epoch_start_time = time.time()

            if self.num_completed_epochs > 0:

                with tf.Session() as sess:

                    self.set_variable_values(sess, self.tf_vars)

                    dh.start_epoch(self.batch_type, self.batch_size)
                    while not dh.is_epoch_finished():
                    
                        (feed_dict, batch_offset) = self.get_next_batch_as_feed_dict(dh, False)
                        
                        if summary_node is None:
                            _ = sess.run(objective, feed_dict=feed_dict)
                        else:
                            (_, summary_output) = sess.run([objective, summary_node], feed_dict=feed_dict)

                    if summary_output is not None:
                        summary_writer.add_summary(summary_output, self.num_remaining_epochs)

                    if self.num_completed_epochs % 10 == 0 or self.num_remaining_epochs == 0:              
                        self.save_variables_to_file(sess, 'epoch_' + str(self.num_completed_epochs))

                    self.tf_vars = self.get_variable_values(sess)

            with tf.Session() as sess:

                if self.num_completed_epochs == 0:
                    self.load_or_init_variables(sess, dh)
                else:
                    self.set_variable_values(sess, self.tf_vars)

                #Evaluate
                (loss, acc, output_activations) = self.test_model(sess, dh, False)
                (val_loss, val_acc, output_activations) = self.test_model(sess, dh, True)

                #Store results
                log['loss'].append(loss)
                log['acc'].append(acc)
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)

                elapsed = int(time.time() - epoch_start_time)

                print("Epoch {0} (-{1}): {2:.3f}% ({3}); {4:.3f}% ({5}) | {6}s".format(self.num_completed_epochs,
                                                                                self.num_remaining_epochs, acc*100, loss, val_acc*100, val_loss, elapsed))

                self.num_completed_epochs += 1
                self.num_remaining_epochs -= 1
#                 if self.num_remaining_epochs == 0:
#                     answer = ""
#                     while not contains_int(answer):
#                         answer = input('How many more epochs do you want to run: ')
#                     self.num_remaining_epochs = int(answer)

        self.write_training_log(log)
    
#             self.result_test, preds = sess.run([self.get_tensor_accuracy(), self.layers['output'].content_tf],
#                                                feed_dict={self.get_tensor_X(): dh.X_test, self.get_tensor_y(): dh.y_test})
#             self.result_test = val_acc
        
        #self.handle_predictions(preds, dh)

        #print("Train accuracy: {0:.2f}".format(self.result_train * 100))
        #print("Test accuracy: {0:.2f}".format(self.result_test * 100))
        
        if self.history != None:
            self.history.write_stub("{0:.2f}.result".format(val_acc * 100))
        
        #self.visuals(sess, dh)

    def visuals_guided_backprop(self, dh, pretested_output_act, num_images):
        """Generate visual output using guided backpropagation.
        Assumes that the caller has switched the current graph to use the guided
        backpropagation relu activation function whereby negative gradients are replaced
        with zero.
        Also assumes that the only trainable values in the graph are those in the input image.
        Params:
            dh: the data handler that contains the images and labels
            pretested_output_act: the activations of the neurons in the output layer as
                determined during a pretest, when provided the original image
            max_num_images: the maximum number of images to show in this one visualization
        Returns:
            None 
        """

        # Create the folder that visuals are written to.
        if self.history is not None:
            visuals_path = self.history.create_folder('visuals')
        
        with tf.Session() as sess:
            
            # Optimize the graph with the objective being to reduce our specialized cost
            # function on the output layer.  This cost function equates to the negative of the
            # correct score, meaning it is never satisfied and always tries to exaggerate
            # the output score for the correct class.
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
            objective = optimizer.minimize(self.layers["output"].cost2)

            # Extract the gradients that relate the image itself to our cost function.
            grads = tf.gradients(self.layers["output"].cost2, self.layers["image"].content_tf)

            # Create a visualization object.  This is the greater image that all the smaller images
            # are added into.
            vis = visualization3(self.layers_as_list, 96, 24, 24, 12, 800)

            # Tell the data handler to start an epoch for us.
            # Batch size for this visualization is always 1.  Some parts of this function assume
            # this; some do not.
            dh.start_epoch(self.batch_type, self.batch_size, False)

            # While the epoch is not finished...
            while not dh.is_epoch_finished():

                # Get the next batch of images and labels in a TensorFlow-friendly feed dictionary.
                # Also get the offset which is the index of the starting image within the full dataset.
                feed_dict, offset = self.get_next_batch_as_feed_dict(dh, True)

                # Load the weights and biases and other parameters from our pretrained model.
                self.load_or_init_variables(sess, dh)
                
                # Assuming that the prediction is correct, identify the output neuron most activated by
                # the image.
                argmax_x = np.argmax(pretested_output_act[offset], axis=0)

                # Generate a vector (an "overlay") that has a 1 in the position of the correct output,
                # and a zero in all other positions, then shape it appropriately to handle the fact that
                # (in future) batches might contain more than one image.
                pretested_output_overlay = np.zeros((1, pretested_output_act[offset].shape[1]))
                pretested_output_overlay[0, argmax_x] = 1.0
                np.expand_dims(pretested_output_overlay, 0)

                # Add this overlay into the feed dictionary we will send to TensorFlow.
                feed_dict[self.layers["output"].pretested_output_overlay] = pretested_output_overlay

                # Initialize lists of image tiles for the updated images, and abs.value, positive, and
                # negative gradients.
                image_tiles = []
                grad_neg_tiles = []
                grad_pos_tiles = []
                grad_abs_tiles = []

                # Assign the image variable from the TensorFlow placeholder.  This is pulled from the feed dictionary.
                sess.run(self.layers["image"].assign_op, feed_dict)

                # Re-extract the image from TensorFlow and add it to the visualization to prove correct initialization.
                # Create spacers for the other tiles in the column.
                im = sess.run(self.layers["image"].content_tf)
                image_tiles.append(im[0])
                grad_abs_tiles.append(np.zeros((28,28,1)))
                grad_neg_tiles.append(np.zeros((28,28,1)))
                grad_pos_tiles.append(np.zeros((28,28,1)))

                # Refine the original image a large number of times, capturing the gradients and updating
                # the visualization at intervals.
                for iter in range(2400):
                    
                    # This is the main instruction to TensorFlow.   Optimize based on our objective.
                    sess.run(objective, feed_dict)

                    if (iter % 400 == 0):

                        # Extract the current cost, gradients, and updated image from the graph. 
                        _, cost, g, im = sess.run([objective, self.layers["output"].cost2, grads, self.layers['image'].content_tf], feed_dict)
                        
                        # Add to our tiles lists the image, the absolute values of the gradients, the
                        # above-zero gradients, and the negation of the below-zero gradients.
                        image_tiles.append(im[0])
                        grad_abs_tiles.append(np.abs(g[0][0]))
                        grad_neg_tiles.append(np.abs(np.maximum(0, g[0][0])))
                        grad_pos_tiles.append(np.negative(np.minimum(0, g[0][0])))

                # Stack together each list of image tiles into a higher-dimensional numpy array
                # expected by the visualization object.
                vis.add_row(np.squeeze(np.stack(image_tiles, 3)), 56, False)
                vis.add_row(np.squeeze(np.stack(grad_abs_tiles, 3)), 56, False)
                vis.add_row(np.squeeze(np.stack(grad_neg_tiles, 3)), 56, False)
                vis.add_row(np.squeeze(np.stack(grad_pos_tiles, 3)), 56, False)

                # After the maximum number of images we want to show has been reached, finalize
                # the visualization, show it, save it to file, and exit the loop and the function.
                if (offset == num_images - 1):
                    v = vis.finalize()
                    v.show()
                    path = os.path.join(visuals_path, 'output.png')
                    v.save(path)
                    return

    def visuals(self, sess, dh):
    
        if self.history is not None:
            visuals_path = self.history.create_folder('visuals')

        conv_layers = [layer for layer in self.layers_as_list if isinstance(layer, lyr.conv_layer)]
        
        batch_size = 100
        for batch_start in range(0, dh.X_train.shape[0], batch_size): #dh.X_train.shape[0]):
            print("Batch start ", batch_start)
            feed_dict = {self.get_tensor_X():dh.X_train[batch_start:batch_start+batch_size]}
             
            layer_tensors = [layer.content_tf for layer in self.layers_as_list]
            weight_tensors = [layer.weights for layer in self.layers_as_list if layer.weights is not None]
            #names_of_layers_with_weights = [layer.name for layer in self.layers_as_list if layer.weights is not None]
            
            #TODO: remove weight tensors?
            tensor_values = sess.run(layer_tensors + weight_tensors, feed_dict=feed_dict)
            conv_tensors = tensor_values[:len(layer_tensors)]
             
            for (layer1, tv) in zip(self.layers_as_list, tensor_values[:len(layer_tensors)]):
                layer1.content = my_array(tv, ['image_num', 'x', 'y', 'oc'])
                 
            for cl in conv_layers:
                cl.handle_activations_for_batch(batch_start)
 
        vis3 = visualization3(self.layers_as_list, 96, 24, 24, 12, 800)
        for cl in conv_layers:
            vis3.add_text_row(cl.name + ' - most activating source features')
            for oc in cl.max_activations_per_output_channel:
                tiles = []
                for act in oc.activations:
                    tiles.append(np.squeeze(act.tile))
                if len(tiles) > 0:
                    #TODO  This is where we would deal with RBG input image
                    tiles2 = my_array(tiles, ['x','y','img'], join_on_axis='img')
                    vis3.add_row(tiles2.arr, tiles[0].shape[0], normalize_across_images=False)
        output = vis3.finalize()
        output.show()
        if self.history is not None:
            path = os.path.join(visuals_path, 'max_act_source_images.png')
            output.save(path)
        vis3 = visualization3(self.layers_as_list, 96, 24, 24, 12, 800)

        for cl in conv_layers:
            vis3.add_text_row(cl.name + ' - blended maximum activation')
            tiles = []
            for oc in cl.max_activations_per_output_channel:
                tiles.append(np.squeeze(oc.generate_blended_tile()))
            tiles2 = my_array(tiles, ['x','y','img'], join_on_axis='img')
            vis3.add_row(tiles2.arr, tiles[0].shape[0]*3, normalize_across_images=False)
        output = vis3.finalize()
        output.show()
        if self.history is not None:
            path = os.path.join(visuals_path, 'max_act_blended_images.png')
            output.save(path)
        vis3 = visualization3(self.layers_as_list, 96, 24, 24, 12, 800)

        print("DONE!!!")
        quit()
            
#             #TODO: get these directly from the layers.  no need then to have a names list
#             layer_activations = tensor_values[:len(layer_tensors)]
#             layer_weights = dict(zip(names_of_layers_with_weights, tensor_values[len(layer_tensors):]))
# 
#             vis = visualization(96, 24, 24, 12, 800)
#             if image_num < 10:
#                 vis.visualize(self.layers_as_list, layer_activations, layer_weights, False)
#                 if self.history is not None:
#                     path = os.path.join(visuals_path, 'output' + str(image_num) + '.png')
#                     vis.output.save(path)
#                 vis.output.show()
# 
#             vis2.visualize(self.layers_as_list, layer_activations, layer_weights, ((image_num+1) % 50) == 0)
#             
#             if image_num % 50 == 0: print("Image ", image_num)
#         
#         vis2_final = vis2.finalize()
#         if self.history is not None:
#             path = os.path.join(visuals_path, 'max_activations.png')
#             vis2_final.save(path)
#         vis2_final.show()
    
#     def new_visuals(self, sess, dh):
#  
#         vis2 = visualization2(self.layers_as_list, 96, 24, 24, 12, 800)
#          
#         for item in range(3): #dh.X_train.shape[0]):
#             feed_dict = {self.get_tensor_X():dh.X_train[item:item+1]}
#              
#             layer_tensors = [layer.content_tf for layer in self.layers_as_list]
#             weight_tensors = [layer.weights for layer in self.layers_as_list if layer.weights is not None]
#             names_of_layers_with_weights = [layer.name for layer in self.layers_as_list if layer.weights is not None]
#              
#             tensor_values = sess.run(layer_tensors + weight_tensors, feed_dict=feed_dict)
#  
#             #TODO: get these directly from the layers.  no need then to have a names list
#             layer_activations = tensor_values[:len(layer_tensors)]
#             layer_weights = dict(zip(names_of_layers_with_weights, tensor_values[len(layer_tensors):]))
#      
#             if self.history is not None:
#                 visuals_path = self.history.create_folder('visuals')
#  
#             vis = visualization(96, 24, 24, 12, 800)
#             if item < 10:
#                 vis.visualize(self.layers_as_list, layer_activations, layer_weights, False)
#                 if self.history is not None:
#                     path = os.path.join(visuals_path, 'output' + str(item) + '.png')
#                     vis.output.save(path)
#                 vis.output.show()
#  
#             vis2.visualize(self.layers_as_list, layer_activations, layer_weights, ((item+1) % 50)==0)
#              
#             if item % 50 == 0: print(item)
#          
#         vis2_final = vis2.finalize()
#         if self.history is not None:
#             path = os.path.join(visuals_path, 'max_activations.png')
#             vis2_final.save(path)
#         vis2_final.show()
         
    def get_next_batch_as_feed_dict(self, dh, use_guided_relu):

        (batch_offset, batch_X, batch_y, batch_Z) = dh.get_next_batch()
        
        feed_dict = {self.get_tensor_X():batch_X, self.get_tensor_y():batch_y}
        
        if self.get_tensor_Z() is not None:
            feed_dict[self.get_tensor_Z()] = batch_Z

        return (feed_dict, batch_offset)

#     def visuals(self, sess, dh):
# 
#         conv1 = self.layers['conv1']
#         
#         dh.start_epoch(batch_type.MINI, self.batch_size)
# 
#         best_scores = None
#         best_image_ids = None
#         
#         while not dh.is_epoch_finished():
#      
#             batch_offset = dh.batch_offset    
#             feed_dict = self.get_next_batch_as_feed_dict(dh)
# 
#             (scores, image_ids) = sess.run([conv1.top_k_scores, conv1.top_k_image_ids], feed_dict=feed_dict)
#             
#             if best_scores is None:
#                 best_scores = scores
#                 best_image_ids = image_ids
#             else:
#                 best_scores = np.append(best_scores, scores, axis=1)
#                 best_image_ids = np.append(best_image_ids, image_ids + batch_offset, axis=1)
# 
#         indexes = np.argsort(best_scores, axis=1)
#         indexes = np.flip(indexes, axis=1)
#         indexes = indexes[:,0:conv1.visualizations_top_k]
#         
#         new_best_scores = np.zeros((len(indexes), conv1.visualizations_top_k))
#         new_best_image_ids = np.zeros((len(indexes), conv1.visualizations_top_k), dtype=int)
#         
#         for i in range(len(indexes)):
#             new_best_scores[i] = best_scores[i, indexes[i]]
#             new_best_image_ids[i] = best_image_ids[i, indexes[i]]
#     
#         return
#         if self.history is not None:
#             path = self.history.create_folder('filter_visuals')
#             filter_extent = conv1.filter_extent
#             w = conv1.weights.eval()
#             b = conv1.biases.eval() #unused
#             slices = conv1.extent - filter_extent + 1
#             excitations = np.zeros((len(indexes), filter_extent, filter_extent, 1))
#             for filter in range(len(indexes)):
#                 w_filter = np.squeeze(w[:,:,:,filter])
#                 for k in range(conv1.visualizations_top_k):
#                     image_id = new_best_image_ids[filter, k]
#                     image = np.squeeze(dh.X_train[k])
#     #             for k in range(dh.num_train_samples):
#                     #imsave(os.path.join(path, str(filter) + '_' + str(k) + '.png'), np.squeeze(image))
#                     max_sum_exc = 0
#                     for x in range(slices):
#                         for y in range(slices):
#                             image_part = image[x:x+filter_extent, y:y+filter_extent]
#                             excitation = np.multiply(image_part, w_filter)
#                             sum_exc = np.sum(excitation)
#                             if sum_exc > max_sum_exc:
#                                 max_sum_exc = sum_exc
#                                 max_exc = excitation
#                 #max_exc = np.squeeze(max_exc)
#                 max_exc = max_exc - max_exc.min()
#                 max_exc = max_exc / max_exc.max()
#                 max_exc *= 255
#                 max_exc = np.uint(max_exc)
#             
#                 imsave(os.path.join(path, 'e' + str(filter) + '.png'), max_exc)
#                 imsave(os.path.join(path, 'w' + str(filter) + '.png'), w_filter)
            
    def test_model(self, sess, dh, use_validation_set):
        assert self.g is tf.get_default_graph()
        total_loss, total_acc, num_batches = 0, 0, 0
        all_output_activations = []

        dh.start_epoch(self.batch_type, self.batch_size, use_validation_set)
        while not dh.is_epoch_finished():

            (feed_dict, batch_offset) = self.get_next_batch_as_feed_dict(dh, False)

            #batch_loss, batch_acc = sess.run([self.get_tensor_cost(), self.get_tensor_accuracy()], feed_dict=feed_dict)
            batch_loss, batch_acc, batch_output_activations = sess.run([self.get_tensor_cost(), self.get_tensor_accuracy(), self.layers['output'].content_tf], feed_dict=feed_dict)

            total_loss += batch_loss
            total_acc += batch_acc
            all_output_activations.append(batch_output_activations)
            num_batches += 1
           
        if num_batches > 0:
            acc = total_acc / num_batches
        
        return total_loss, acc, all_output_activations

    def handle_predictions(self, preds, dh):
        
        if self.history != None:

            false_pos_path = self.history.create_folder("false_pos")
            false_neg_path = self.history.create_folder("false_neg")
            true_pos_path = self.history.create_folder("true_pos")
            true_neg_path = self.history.create_folder("true_neg")
            
            for (pred_, y_, file) in zip(preds, dh.y_test, dh.files_test):
                if pred_[1] > pred_[0]:
                    if y_[1] > y_[0]:
                        shutil.copy(file, true_pos_path)
                    else:
                        shutil.copy(file, false_pos_path)
                else:
                    if y_[0] < y_[1]:
                        shutil.copy(file, true_neg_path)
                    else:
                        shutil.copy(file, false_neg_path)

    def save_variables_to_file(self, sess, folder_name):

        if self.history != None:
            
            path = self.history.create_folder("vars")
            path = os.path.join(path, folder_name)
            os.mkdir(path)
            path = os.path.join(path, "vars.ckpt")
            
            saver = tf.train.Saver(self.get_savable_variables())
            saver.save(sess, path)

    def write_training_log(self, log):

        if self.history != None:
            
            self.history.write_dictionary_to_csv("training.csv", log)

    def plot_training_log(self, log):
        
        if self.history != None:
    
            epochs =range(len(log['acc']))
            
            pl.subplot(2, 1, 1)
            pl.plot(epochs, log['acc'],'r', label='train')
            pl.plot(epochs, log['val_acc'],'b', label='test')
            pl.title('Accuracy')
            pl.xlabel('epoch')
            pl.legend(loc='lower right')
            
            pl.subplot(2, 1, 2)
            pl.plot(epochs, log['loss'],'r', label='train')
            pl.plot(epochs, log['val_loss'],'b', label='test')
            pl.title('Loss')
            pl.xlabel('epoch')
            pl.legend(loc='upper right')
            
            pl.savefig(self.history.get_absolute_path("plot.jpg"))

    def get_variable_values(self, sess):
        variables = tf.global_variables()
        keys = [v.name for v in variables]
        values = [v.eval() for v in variables]
        d = dict(zip(keys, values))
        return d

    def set_variable_values(self, sess, d):
        variables = tf.global_variables()
        for key in d.keys():
            v = [v for v in variables if v.name == key][0]
            sess.run(v.assign(d[key]))