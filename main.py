# Author: Matthew Sudmann-Day

from history import *
from data_handler import *
from model import *
from misc import *
from params import *
import inception

params = msd_params()
params.execution_mode = execution_mode.TRAINING
params.data_source = data_source.MNIST
params.train_ratio = 0.75
params.activation_function = activation_function.RELU_TENSORFLOW_BUILTIN
params.batch_normalization_method = batch_normalization_method.TENSORFLOW_BUILTIN
params.batch_size = 24
params.batch_type = batch_type.MINI
params.dropout_ratio_conv = 0.25
params.dropout_ratio_fc = 0.25
params.learning_rate = 0.0005
params.loss_function = loss_function.SOFTMAX_CROSS_ENTROPY
params.pooling_method = pooling_method.MAX
params.max_inputs = 24*12
params.num_epochs = 10
params.optimization_method = optimization_method.ADAM
params.shuffle_data = True
params.shuffle_data_seed = 12345

visualization = False

with history(params) as hist:

    print_versions()

    dh = data_handler(params)
    dh.load_data()
    #dh.merge_data(inception.get_activations('mixed_6/join'))
    dh.split_data()

    if visualization:
        
        # Create the model built on TensorFlow (False = not for visualization purposes).
        model = model_tf(params, hist)
        model.create_model(dh, False)
        
        # Reload a pretrained model and extract output activations for the entire dataset.
        with tf.Session(model.g) as sess:
            model.load_or_init_variables(sess, dh)
            _, __, pretested_output_act = model.test_model(sess, dh, False)
            
        # Recreate the model with different variables established as "training variables".
        # Also this results in a different activation function.
        model = model_tf(params, hist)
        model.create_model(dh, True)
        
        # Generate visuals using guided backpropagation for the first three images in the dataset.
        model.visuals_guided_backprop(dh, pretested_output_act, 3)
    else:
        model = model_tf(params, hist)
        with model.g.as_default():
            model.create_model(dh, False)
            model.train_normal(dh)

print("done")

#dh.Z_all = inception.extract_activations_from_model('mixed_7/tower_1/conv_4') 
#dh.Z_all = inception.extract_activations_from_model('mixed_7/tower_2/pool') 
#dh.Z_all = inception.get_activations('pool_3')
