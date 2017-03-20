print("Ignore this:")
from keras.datasets import mnist
import numpy as np

def load_data():
    
    # The mnist package splits up our training and test sets for us.  I don't like that.
    # So I merge them back together into simply X and y and make the split where I want.
    (X_1, y_1), (X_2, y_2) = mnist.load_data()
    X = np.concatenate((X_1, X_2), axis=0)
    X = np.expand_dims(X, 3)
    y = np.concatenate((y_1, y_2))

    return (X, y)
