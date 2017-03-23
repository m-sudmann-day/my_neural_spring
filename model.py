import sys
import platform
import numpy as np
import math
import tensorflow as tf
import cv2
import os
from glob import glob
import random
import csv
import shutil

import matplotlib
matplotlib.use('Agg')
import pylab as pl

class model:

    def __init__(self, seed=0, history_path=None):

        if seed > 0:
            np.random.seed(seed)
            
        if history_path != None:
            self.history_path = history_path

    def load_data(self, train_ratio=0.75):

        self.train_ratio = train_ratio

        loaded_images = list()
        labels = list()
        images = glob('../images/1*')
        images.extend(glob('../images/2*'))
        
        random.shuffle(images)
        
        images = images[:100]
        
        for file in images:
            
            filename = os.path.basename(file)
            label = int(filename[0])
            labels.append(label)
            
            im = cv2.imread(file)
            
            im = np.swapaxes(im, 1, 2)
            im = np.swapaxes(im, 0, 1)
            loaded_images.append(im)

        X = np.stack(loaded_images)
        y = np.array(labels)

        # The code assumes the lowest label is a zero.
        y -= min(labels)
  
        self.num_image_layers = X.shape[1]
        
        # The mnist package splits up our training and test sets for us.  I don't like that.
        # So I merge them back together into simply X and y and make the split where I want.
        #(X_1, y_1), (X_2, y_2) = mnist.load_mnist_data()
        #X = np.concatenate((X_1, X_2), axis=0)
        #y = np.concatenate((y_1, y_2))

        # Set the number of classes.  Okay, it's always 10 for this data.
        self.num_classes = len(set(y))

        #X, y = self.process_images(X, y)

        X = self.prepare_inputs(X)

        # Calculations for train and test sizes based on train_ratio.
        num_available_samples = y.shape[0]
        num_train_samples = math.floor(num_available_samples * self.train_ratio)
        num_test_samples = num_available_samples - num_train_samples

        # Get the train and test sets, X (images) and y (labels).
        self.X_train = X[:num_train_samples]
        self.X_test = X[num_train_samples:(num_train_samples + num_test_samples)]
        self.y_train = y[:num_train_samples]
        self.y_test = y[num_train_samples:(num_train_samples + num_test_samples)]
        self.images_train = images[:num_train_samples]
        self.images_test = images[num_train_samples:(num_train_samples + num_test_samples)]

        # Reshape the data so that they are dimensioned like real images (layer/X/Y).
        # I might have x and y coordinates reversed, but they both happen to be the same: 28
        shape = self.X_train.shape
        self.num_pixels_X = shape[len(shape) - 2]
        self.num_pixels_Y = shape[len(shape) - 1]
        self.num_pixels = self.num_pixels_X * self.num_pixels_Y
        self.X_train = self.X_train.reshape(num_train_samples, self.num_image_layers, self.num_pixels_X, self.num_pixels_Y)
        self.X_test = self.X_test.reshape(num_test_samples, self.num_image_layers, self.num_pixels_X, self.num_pixels_Y)

        # Preserve some descriptive numbers.
        self.num_available_samples = num_available_samples
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

    def prepare_inputs(self, X):

        # Convert to float32 to match the expectation of the input layer.
        X = X.astype('float32')

        # Pixel values range in [0, 255].  Translate into [-1, 1].
        # Instead of doing this on the entire set at once, it is done individually
        # on each image so that those images that do not use the full range of
        # input values get their values stretched to fill the range.  This could
        # help accommodate different lighting conditions.  For an RGB image, this
        # would have to be done across the three layers, not separately for each layer.
        for i in range(X.shape[0]):
            one_image = X[i]
            max = one_image.max()
            min = one_image.min()
            X[i] = 2 * ((one_image - min) / (max - min)) - 1
        return X

    def encode_outputs(self):

        # Use one-hot encoding, i.e. "categorical".
        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def create_model(self):

        input_shape = (self.num_image_layers, self.num_pixels_X, self.num_pixels_Y)
        model = Sequential()

        model.add(Convolution2D(64, 7, 7, subsample=(1, 1), border_mode='same', input_shape=input_shape, activation='relu'))
        model.add(Convolution2D(64, 5, 5, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
 
        model.add(Convolution2D(128, 5, 5, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(Convolution2D(128, 5, 5, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
 
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
        for layer in model.layers:
            print(layer.input_shape, layer.output_shape)
        self.model = model
    
    def write_learning_log(self):

            log_path = os.path.join(self.history_path, "learning.csv")
            log = self.learning_log.history
            log_len =len(log['acc'])
            with open(log_path, 'w') as f:
                writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['acc', 'loss', 'val_acc', 'val_loss'])
                for i in range(log_len):
                    writer.writerow([log['acc'][i], log['loss'][i], log['val_acc'][i], log['val_loss'][i]])

            path = os.path.join(self.history_path, "model.json")
            with open(path, 'w') as f:
                f.write(self.model.to_json())
    
    def plot_learning_log(self):
        
        plot_path = os.path.join(self.history_path, "plot.jpg")
        log = self.learning_log.history
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
        
        pl.savefig(plot_path)

    def train(self, num_epochs=10, batch_size=50):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.iterations_per_epoch = self.num_train_samples / self.batch_size

        self.encode_outputs()
        self.create_model()

        self.learning_log = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                                 nb_epoch=num_epochs, batch_size=batch_size, verbose=2)

        if self.history_path != None:
            self.write_learning_log()
            self.plot_learning_log()

    def test(self, use_training_set=False):
        
        if (use_training_set):
            scores = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        else:
            scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            
        return (scores[1])
    
    def predict(self):
        if self.history_path != None:
            false_pos_path = os.path.join(self.history_path, "false_pos")
            false_neg_path = os.path.join(self.history_path, "false_neg")
            true_pos_path = os.path.join(self.history_path, "true_pos")
            true_neg_path = os.path.join(self.history_path, "true_neg")
            
            os.mkdir(false_pos_path)
            os.mkdir(false_neg_path)
            os.mkdir(true_pos_path)
            os.mkdir(true_neg_path)

            preds = self.model.predict(self.X_test)
            
            for (pred, y, image) in zip(preds, self.y_test, self.images_test):
                if pred[1] > pred[0]:
                    if y[1] > y[0]:
                        shutil.copy(image, true_pos_path)
                    else:
                        shutil.copy(image, false_pos_path)
                else:
                    if y[0] < y[1]:
                        shutil.copy(image, true_neg_path)
                    else:
                        shutil.copy(image, false_neg_path)

#    def load_model(self, path):
#        with open(path, 'r') as json_file:
#            json = json_file.read()
#        self.model = keras.models.model_from_json(json)
#        self.model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
