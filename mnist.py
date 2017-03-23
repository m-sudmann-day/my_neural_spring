import os
import numpy as np
import shutil
import requests
import gzip
from six.moves import cPickle

def load_data():

    filename = 'mnist.pkl.gz'
    remote_folder = 'https://s3.amazonaws.com/img-datasets/'
    local_folder = '../all_images/mnist/'

    remote_path = os.path.join(remote_folder, filename)
    local_path = os.path.join(local_folder, filename)

    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    if not os.path.exists(local_path):
        response = requests.get(remote_path, stream=True)
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        del response

    f = gzip.open(local_path, 'rb')
    (X_1, y_1), (X_2, y_2) = cPickle.load(f, encoding='bytes')

    X = np.concatenate((X_1, X_2), axis=0)
    X = np.expand_dims(X, 3)
    y = np.concatenate((y_1, y_2))

    return(X, y)
