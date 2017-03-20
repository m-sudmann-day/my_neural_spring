import numpy as np
from scipy.misc import imsave
from PIL import Image, ImageDraw, ImageFont
from misc import *
from my_array import *
import layers as lyr

class visualization2: 
        
    def __init__(self, layers_as_list, activation_tile_size, weight_tile_size, indiv_neuron_tile_size_large, indiv_neuron_tile_size_small, max_width):
        self.activation_tile_size = activation_tile_size
        self.weight_tile_size = weight_tile_size
        self.indiv_neuron_tile_size_large = indiv_neuron_tile_size_large
        self.indiv_neuron_tile_size_small = indiv_neuron_tile_size_small
        self.margin_horiz = 5
        self.text_margin_below = 5
        self.text_margin_above = 3
        self.max_width = max_width
        self.frame_edge_thickness = 1
        self.y = self.text_margin_above
        self.width = self.margin_horiz
        self.height = self.text_margin_above
        self.output = np.zeros((self.width, self.height), dtype=np.uint8)
        self.layers_as_list = layers_as_list
        self.max_acts_dict = dict()
        self.sum_acts_dict = dict()
        self.score_dict = dict()
        for layer in layers_as_list:
            if isinstance(layer, lyr.conv_layer):
                self.max_acts_dict[layer] = np.zeros((layer.filter_extent, layer.filter_extent, layer.num_output_channels, layer.num_input_channels), dtype=np.float32)
                self.sum_acts_dict[layer] = np.zeros((layer.filter_extent, layer.filter_extent, layer.num_output_channels, layer.num_input_channels), dtype=np.float32)
                self.score_dict[layer] = np.zeros((layer.num_output_channels, layer.num_input_channels), dtype=np.float32)
    
    def handle_maximum_activations_for_one_filter(self, current_layer, filter_weights, output_activations_one_channel, input_activations):
        
        filter_extent = current_layer.filter_extent
        input_extent = current_layer.input_layer.extent
        padding = int((filter_extent - 1) / 2)
        
        (max_x, max_y) = np.unravel_index(output_activations_one_channel.arr.argmax(), output_activations_one_channel.shape)
        
        all_max_activations = list()
        
        for input_channel in range(input_activations.shape_dict['oc']):
            
            padded_input_activations_one_channel = input_activations.slice('oc', input_channel)
            filter_weights_for_input_channel = filter_weights.slice('ic', input_channel)

            temp_image = np.zeros((current_layer.extent + padding*2, current_layer.extent + padding*2))
            temp_image[padding:input_extent+padding, padding:input_extent+padding] = padded_input_activations_one_channel.arr
            padded_input_activations_one_channel = my_array(temp_image, padded_input_activations_one_channel.axes)
            
            max_activations = padded_input_activations_one_channel.arr[max_x:max_x+filter_extent, max_y:max_y+filter_extent]
            all_max_activations.append(max_activations)

       #print("X",len(all_max_activations))
        #all_max_activations = my_array(all_max_activations, ['x', 'y', 'oc'], join_on_axis='oc')       
        return all_max_activations

    def visualize(self, layers, layer_activations, layer_weights_dict, reset):
        
        image_num = 0
        previous_layer_activations = None
        
        # loop through the layers in the model and the activations for each layer extracted from TensorFlow
        for current_layer, current_layer_activations in zip(layers, layer_activations):

            individual_neurons = (isinstance(current_layer, lyr.fc_layer) or isinstance(current_layer, lyr.output_layer))

            if isinstance(current_layer, lyr.conv_layer):
                
                weights = layer_weights_dict[current_layer.name]
                weights = my_array(layer_weights_dict[current_layer.name], ['x','y','ic','oc'])
                
                output_activations = my_array(current_layer_activations, ['image', 'x', 'y', 'oc'])
                output_activations = output_activations.slice('image', 0)
                
                filter_extent = current_layer.filter_extent
                output_extent = current_layer.extent
                input_extent = previous_layer.extent
                input_activations = my_array(previous_layer_activations, ['image', 'x', 'y', 'oc'])
                num_output_channels = current_layer.num_output_channels
                
                #row_images = np.zeros((filter_extent, filter_extent, current_layer.num_output_channels))

                for (filter_weights, activations_by_oc, i) in zip(weights.slices('oc'), output_activations.slices('oc'), range(num_output_channels)):
                    input_activations_for_one_image = input_activations.slice('image', image_num)
                    max_act = self.handle_maximum_activations_for_one_filter(current_layer, filter_weights, activations_by_oc,
                                                                   input_activations_for_one_image)
                    
                    for j in range(len(max_act)):

                        score = max_act[j].max()
                        if score > self.score_dict[current_layer][i,j]:
                            self.score_dict[current_layer][i,j] = score
                            self.max_acts_dict[current_layer][:,:,i,j] = max_act[j]
                    
                        if reset:
                            self.sum_acts_dict[current_layer][:,:,i,j] += self.max_acts_dict[current_layer][:,:,i,j]
                            self.max_acts_dict[current_layer].fill(0)
                            self.score_dict[current_layer].fill(0)
                        
            previous_layer = current_layer
            previous_layer_activations = current_layer_activations
        
    def finalize(self):

        self.increase_height(self.margin_horiz)

        for layer in self.layers_as_list:
            if isinstance(layer, lyr.conv_layer):
                sum_acts = self.sum_acts_dict[layer]
                
                self.add_text_row(layer.name + ' maximum activations')
    
                for j in range(sum_acts.shape[3]):
                    temp = sum_acts[:,:,:,j]
                    self.add_row(temp, self.weight_tile_size, normalize_across_images=False)

        result = np.copy(self.output)
        result = np.swapaxes(result, 0, 1)
        result = Image.fromarray(result).convert('L')
        return(result)

    def ensure_width(self, width):
        if width > self.width:
            self.output = np.pad(self.output, ((0, width - self.width), (0, 0)), mode='constant')
            self.width = width
    
    def increase_height (self, additional_height):
        self.output = np.pad(self.output, ((0, 0), (0, additional_height)), mode='constant')
        self.height += additional_height

    def add_text_row(self, text):
        
        fnt = ImageFont.truetype('FreeMono.ttf', 18)
        (text_width, text_height) = fnt.getsize(text)
        canvas = Image.new('L', (text_width, text_height), 'black')
        draw = ImageDraw.Draw(canvas)
        draw.text((0, 0), text, fill=(255), font=fnt)
        text_image = np.swapaxes(np.asarray(canvas), 0, 1)
        
        self.x = self.margin_horiz
        self.ensure_width(self.x + text_width + self.margin_horiz)
        self.increase_height(text_height + self.text_margin_above + self.text_margin_below)
        self.y += self.text_margin_above
        self.output[self.x:self.x + text_width, self.y:self.y + text_height] = text_image
        self.y += text_height + self.text_margin_below
        
    def add_row(self, images, tile_size, normalize_across_images=False):

        images = np.copy(images) #because we normalize it, changing it in place
        self.x = self.margin_horiz
        self.increase_height(tile_size + self.frame_edge_thickness * 2)
        
        # Some layers visualize better normalized across the whole set.  Others are normalized individually.
        if normalize_across_images:
            images = normalize_uint8_255(images, 2)
        
        for image_num in range(images.shape[2]):
            
            image = images[:,:,image_num]
            if not normalize_across_images:
                image = normalize_uint8_255(image, 1.5)
                
            # Make a true image from our array and resize it.
            #image = np.squeeze(image)
            image = Image.fromarray(image).convert('L')
            image = image.resize((tile_size, tile_size))
            image = image.transpose(Image.ROTATE_90)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Create a white box larger than the images by frame_edge_thickness.  Then project the image into
            # this box just leaving the white frame border.
            #TODO: should be able to do this with np.pad when it's fixed
            frame = np.ones([tile_size + self.frame_edge_thickness * 2, tile_size + self.frame_edge_thickness * 2], dtype=np.uint8) * 255
            frame[self.frame_edge_thickness:tile_size + self.frame_edge_thickness,
                  self.frame_edge_thickness:tile_size + self.frame_edge_thickness] = image
            image = frame

            pad = tile_size + self.frame_edge_thickness * 2
            self.ensure_width(self.x + pad + self.margin_horiz)

            self.output[self.x:self.x + pad, self.y:self.y + pad] = image
            self.x += pad

        self.y += tile_size + self.frame_edge_thickness * 2
