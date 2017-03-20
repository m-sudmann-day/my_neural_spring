import numpy as np
from activation import *
from misc import *

class max_activation_set():

    def __init__(self, layer, use_top_k_images):
        
        self.activations = list()
        self.layer = layer
        self.use_top_k_images = use_top_k_images
        
    def handle_activations(self, content, first_image_num):
        
        image_num = first_image_num
        for image_content in content.slices('image_num'):
            
            max_value = image_content.arr.max()
            (max_x, max_y) = np.unravel_index(image_content.arr.argmax(), image_content.shape)

            new_activation = activation(max_x, max_y, max_value, self.layer, image_num)
            self.activations.append(new_activation)
       
            image_num += 1
            
            for z in self.activations:
                if (z.layer.name != self.layer.name):
                    print("BBBB")
    
    def sort_and_truncate(self):
        
        for z in self.activations:
            if (z.layer.name != self.layer.name):
                print("CCCC")

        self.activations = sorted(self.activations, key = lambda a:a.value, reverse=True)
        if len(self.activations) > self.use_top_k_images:
            self.activations = self.activations[:self.use_top_k_images]
    
        for z in self.activations:
            if (z.layer.name != self.layer.name):
                print("DDDD")

    def calculate_image_coordinates(self):
        
        for z in self.activations:
            if (z.layer.name != self.layer.name):
                print("EEEE")

        for activation in self.activations:
            activation.calculate_image_coordinates()

    def get_tiles(self, first_image_num_in_batch):
        
        for z in self.activations:
            if (z.layer.name != self.layer.name):
                print("UUUU")

        for activation in self.activations:
            if activation.tile is None:
                activation.get_tile(first_image_num_in_batch)

    def generate_blended_tile(self):

        if len(self.activations) == 0:
            return None
        
        blended_tile = np.zeros(self.activations[0].tile.shape, dtype=float)
        
        for activation in self.activations:
            blended_tile = blended_tile + (activation.tile * activation.value * activation.value)
        
        return(blended_tile)