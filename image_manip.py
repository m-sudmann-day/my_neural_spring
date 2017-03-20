from PIL import Image, ImageFilter
import os
from glob import glob
from skimage import color
import cv2
from scipy import ndimage
from scipy import misc
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import shutil
import sys
import io
import random

image_read_mask="../all_images/2_labeled/"
image_write_path="../images/"
new_shape = (136,136)

for f in glob(os.path.join(image_write_path, "*")):
    os.remove(f)

for label in (1,2):
    
    for file in glob(image_read_mask + str(label) + "/*"):
        
        filename = os.path.basename(file)
        root = image_write_path + str(label)
        
        print(filename)
        
        im = Image.open(file)
        
        if (im.width < im.height):
            im = im.transpose(Image.ROTATE_90)
    
        if (im.width > im.height):
            margin = (im.width - im.height) // 2
            im = im.crop((margin, 0, im.height + margin, im.height))

        im = im.resize(new_shape)
        im.save(root + str(label) + "-0" + filename)

        for i in range(3):
            imR = im.rotate(random.randint(1, 17) * 20)
            imR.save(root + str(label) + "-" + str(i) + filename)

        mirror = im.transpose(Image.FLIP_LEFT_RIGHT)
        mirror.save(root + str(label) + "M0" + filename)
        
        for i in range(3):
            mirrorR = im.rotate(random.randint(1, 17) * 20)
            mirrorR.save(root + str(label) + "M" + str(i) + filename)
        
print("done")