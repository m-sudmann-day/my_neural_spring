import tensorflow as tf
from params import *

def relu(tensor, params):

    if params.execution_mode == execution_mode.GUIDED_BACKPROP_VISUALIZATION:

        # TODO: work on any graph?
        with tf.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
            return tf.nn.relu(tensor, name='guided_relu')

    else:

        return tf.nn.relu(tensor, name='relu')
    