from history import *
from data_handler import *
from model_tf import *
from misc import *
import inception

visualization = False
if visualization:
    data_handler_params = {'train_ratio':0.75, 'max_images':40}
    model_params = {'num_epochs':100, 'learning_rate':0.0005, 'dropout':0, 'batch_size':1, 'batch_type':'mini' } # high LR = 0.0005
else:
    data_handler_params = {'train_ratio':0.8, 'max_images':None}
    model_params = {'num_epochs':10000, 'learning_rate':0.00005, 'dropout':0.25, 'batch_size':512, 'batch_type':'mini' } # mnist:0.000005

with history() as hist:

    #hist.write_text_file("notes.txt", notes)
    print("Data loader params:", data_handler_params)
    print("Model params:", model_params)
    print_versions()

    dh = data_handler(data_handler_params)
    dh.load_data(data_handler_source.CIFAR10, shuffle=True, shuffle_seed=123)
    #dh.merge_data(inception.get_activations('mixed_6/join'))
    dh.split_data()

    if visualization:
        
        # Create the model built on TensorFlow (False = not for visualization purposes).
        model = model_tf(model_params, hist)
        model.create_model(dh, False)
        
        # Reload a pretrained model and extract output activations for the entire dataset.
        with tf.Session() as sess:
            model.load_or_init_variables(sess, dh)
            _, __, pretested_output_act = model.test_model(sess, dh, False)
            
        # Recreate the model with different variables established as "training variables".
        # Also this results in a different activation function.
        model = model_tf(model_params, hist)
        model.create_model(dh, True)
        
        # Generate visuals using guided backpropagation for the first three images in the dataset.
        model.visuals_guided_backprop(dh, pretested_output_act, 3)
    else:
        model = model_tf(model_params, hist)
        model.create_model(dh, False)
        model.train_normal(dh)

print("done")

#dh.Z_all = inception.extract_activations_from_model('mixed_7/tower_1/conv_4') 
#dh.Z_all = inception.extract_activations_from_model('mixed_7/tower_2/pool') 
#dh.Z_all = inception.get_activations('pool_3')
