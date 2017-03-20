from random import randint
import numpy as np
from PIL import Image

num_images = 1000
shape_extent = 30
image_extent = 96

solid = np.array([[1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]], dtype=np.uint8)
square = np.array([[1,1,1,1,1,1,1], [1,0,0,0,0,0,1], [1,0,0,0,0,0,1], [1,0,0,0,0,0,1], [1,0,0,0,0,0,1], [1,0,0,0,0,0,1], [1,1,1,1,1,1,1]], dtype=np.uint8)
cross = np.array([[1,0,0,0,0,0,1], [0,1,0,0,0,1,0], [0,0,1,0,1,0,0], [0,0,0,1,0,0,0], [0,0,1,0,1,0,0], [0,1,0,0,0,1,0], [1,0,0,0,0,0,1]], dtype=np.uint8)
plus = np.array([[0,0,0,1,0,0,0], [0,0,0,1,0,0,0], [0,0,0,1,0,0,0], [1,1,1,1,1,1,1], [0,0,0,1,0,0,0], [0,0,0,1,0,0,0], [0,0,0,1,0,0,0]], dtype=np.uint8)
slope_up = np.array([[0,0,0,0,0,0,1], [0,0,0,0,0,1,0], [0,0,0,0,1,0,0], [0,0,0,1,0,0,0], [0,0,1,0,0,0,0], [0,1,0,0,0,0,0], [1,0,0,0,0,0,0]], dtype=np.uint8)
slope_down = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]], dtype=np.uint8)

arr = np.zeros((image_extent, image_extent), dtype=np.uint8)
base_shape = np.zeros((shape_extent, shape_extent), dtype=np.uint8)

def show(arr):
    arr *= 255
    img = Image.fromarray(arr, 'L')
    img.show()

square = np.copy(base_shape)
for i in range(shape_extent-1):
    square[0, i] = square[i, 0] = square[shape_extent-1, i] = square[i, shape_extent-1] = 1
    square[1, i] = square[i, 1] = square[shape_extent-2, i] = square[i, shape_extent-2] = 1

cross = np.copy(base_shape)
for i in range(shape_extent-1):
    cross[i, i] = cross[i, shape_extent-1-i] = 1
    cross[i+1, i] = cross[i+1, shape_extent-1-i] = 1

shapes = [cross,square]
shapes = np.multiply(shapes, 255)

def load_images():

    X = np.random.random_integers(low=0, high=127, size=(num_images, image_extent, image_extent))
    #X = np.zeros((num_images, image_extent, image_extent))
    y = np.zeros((num_images), dtype=np.uint8)
    
    all255s = (np.ones((image_extent, image_extent)) * 255)
    all255s = np.ndarray.astype(all255s, dtype=np.uint8)

    for image_id in range(num_images):
        shape_id = randint(0, len(shapes)-1)
        
        x_coord = randint(0, image_extent - shape_extent)
        y_coord = randint(0, image_extent - shape_extent)
        
        X[image_id, x_coord:x_coord+shape_extent, y_coord:y_coord+shape_extent] += shapes[shape_id]
        X = np.minimum(X, all255s)
        y[image_id] = shape_id

        im = np.squeeze(X[image_id])
        im = Image.fromarray(im, 'L').convert('L')
        im.save('../z/'+str(shape_id)+'_'+str(image_id)+".png")
    
    X = np.expand_dims(X, 3)
    X = np.ndarray.astype(X, np.uint8)
    
    return (X, y, None)