import tensorflow as tf
from params import *

def batch_norm(x, is_convolutional, params):

    if params is None:
        bn_method = batch_normalization_method.TENSORFLOW_BUILTIN
    else:
        bn_method = params.batch_normalization_method

    if bn_method == batch_normalization_method.NONE:

        return(x)

    elif bn_method == batch_normalization_method.TENSORFLOW_BUILTIN:

        with tf.name_scope('bn'):

            if is_convolutional:
                axes = [0,1,2]
            else:
                axes = [0]

            gamma = tf.Variable(1.0, name='gamma', dtype=tf.float32)
            beta = tf.Variable(0.0, name='beta', dtype=tf.float32)

            batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')

            return tf.nn.batch_normalization(x,batch_mean,batch_var,beta,gamma,0.0001, name='batch_norm')

    elif bn_method == batch_normalization_method.TRAINABLE:

            running_mean = tf.Variable(0.0, name='running_mean')
            running_var = tf.Variable(0.0, name='running_var')


            # normalize: x2 = (x-mean)/sqrt(var)
            x2 = tf.divide(tf.subtract(x, batch_mean), tf.sqrt(batch_var))

            # scale and offset with learned params: result = (gamma * x2) + beta
            result = tf.add(tf.mul(gamma, x2), beta)

            return(result)
