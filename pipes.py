import numpy as np
import random

def load_images(self):

    images = list()
    labels = list()
    files = glob('../images/1*')
    files.extend(glob('../images/2*'))
    
    random.shuffle(files)
    
    for file in files:
        
        filename = os.path.basename(file)
        label = int(filename[0])
        labels.append(label)
        
        im = cv2.imread(file)
        #im = np.swapaxes(im, 1, 2)
        #im = np.swapaxes(im, 0, 1)
        images.append(im)

    return (images, labels, files)