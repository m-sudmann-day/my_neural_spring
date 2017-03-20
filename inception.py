import os
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from data_handler import *

inception_path = '../inception-2015-12-05'

def get_path_from_node_name(node_name):
    
    return os.path.join(inception_path, node_name.replace('/', '__') + '.npy')

def get_activations(node_name):

    path = get_path_from_node_name(node_name)
    if os.path.isfile(path):
        return np.load(path)
    else:
        return extract_activations_from_model(node_name)
    
def extract_activations_from_model(node_name, save_to_file=False):

    print ("Extracting activations for node " + node_name)
    dh = data_handler({'train_ratio':1, 'max_images':None })
    dh.load_data(data_handler_source.PIPES, shuffle=False)
    dh.split_data()
    
    graph_def = tf.GraphDef()
    with gfile.FastGFile(os.path.join(inception_path,'classify_image_graph_def.pb'), 'rb') as f:
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    with tf.Session() as sess:
        
    #    for n in tf.get_default_graph().as_graph_def().node:
    #        print(n.name)
    
    #     rep_options = ['mixed_10/tower_2/conv', 'mixed_10/tower_2/pool', 'pool_3', 'mixed_9/tower_2/conv', 'mixed_9/tower_2/conv',
    #                    'mixed_8/pool', 'mixed_8/tower_1/conv_3', 'mixed_7/tower_1/conv_4',
    #                    'mixed_7/tower_2/pool', 'mixed_6/join', 'mixed_6/tower_2/conv', 'mixed_5/join', 'mixed_4/join', 'mixed_3/join', 'mixed_5/tower_2/pool', 'mixed_4/tower_2/pool']
    #     
    #     for rep_option in rep_options:
    #         X = dh.X_train[0]
    #         reps = sess.graph.get_tensor_by_name(rep_option + ':0')
    #         reps_content = sess.run([reps], {'DecodeJpeg:0':X})[0]
    #         print(rep_option, reps_content.shape)
        
        node = sess.graph.get_tensor_by_name(node_name + ':0')
        node_content = sess.run([node], {'DecodeJpeg:0':dh.X_train[0]})[0]
        reps_shape = [dh.num_train_samples, node_content.shape[1], node_content.shape[2], node_content.shape[3]]
        reps = np.zeros(reps_shape)
        
        #if reps_shape[1] != 17:
        #    return
        
        for (X, y, i) in zip(dh.X_train, dh.y_train, list(range(dh.num_train_samples))):
            node_content = sess.run([node], {'DecodeJpeg:0':X})[0]
            reps[i,:] = node_content
            if i % 100 == 0:
                print(i)
    
        if save_to_file:
            path = get_path_from_node_name(node_name)
            np.save(path, reps)

        return reps
    
# rep_options = ['mixed_10/tower_2/conv', 'mixed_10/tower_2/pool', 'pool_3', 'mixed_9/tower_2/conv', 'mixed_9/tower_2/conv',
#                         'mixed_8/pool', 'mixed_8/tower_1/conv_3', 'mixed_7/tower_1/conv_4',
#                         'mixed_7/tower_2/pool', 'mixed_6/join', 'mixed_6/tower_2/conv', 'mixed_5/join', 'mixed_4/join', 'mixed_3/join', 'mixed_5/tower_2/pool', 'mixed_4/tower_2/pool']
# for rep_option in rep_options:
#     extract_activations_from_model(rep_option)

#extract_activations_from_model('pool_3', True)
