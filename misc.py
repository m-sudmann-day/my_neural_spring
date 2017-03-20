import sys
import numpy as np
import tensorflow as tf
import platform
from enum import Enum

def print_versions():
    print("OS:", platform.platform())
    print("Python:", sys.version.split(' ')[0])
    print("TensorFlow:", tf.__version__)

# to_categorical()
# stolen from Keras 
# https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
def to_categorical(y, nb_classes=None):
    """Converts class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical

class stream_tee(object):
    
    # Based on https://gist.github.com/327585 by Anand Kunal
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None # Hack!
 
    def __getattribute__(self, name):
        return object.__getattribute__(self, name)
 
    def __getattr__(self, name):
        self.__missing_method_name = name # Could also be a property
        return getattr(self, '__methodmissing__')
 
    def __methodmissing__(self, *args, **kwargs):
            # Emit method call to the log copy
            callable2 = getattr(self.stream2, self.__missing_method_name)
            callable2(*args, **kwargs)
 
            # Emit method call to stdout (stream 1)
            callable1 = getattr(self.stream1, self.__missing_method_name)
            return callable1(*args, **kwargs)

def contains_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def normalize_uint8_255(arr, pwr=1):
    arr = arr.astype(float)
    arr -= arr.min()
    if pwr != 1:
        arr = np.power(arr, pwr)
    m = arr.max()
    if m > 1e-5:
        arr /= m
    arr *= 255
    arr = np.uint8(arr)
    return(arr)
