from enum import Enum
import numpy as np

class my_array:
    
    def __init__(self, source, axes, join_on_axis=None):

        if join_on_axis is None:
            self.arr = source
        else:                
            self.arr = np.stack(source, axis=axes.index(join_on_axis))

        self.axes = axes
        self.shape = self.arr.shape
        self.shape_dict = dict(zip(self.axes, self.arr.shape))
   
    def slice(self, axis, index):
        
        axis_position = self.axes.index(axis)
        new_axis_positions = self.axes[:axis_position] + self.axes[axis_position+1:]

        new_arr = np.take(self.arr, index, axis_position)

        return my_array(new_arr, new_axis_positions)

    def slices(self, axis, start=0, stop=None):
        
        result = list()
        axis_position = self.axes.index(axis)
        if stop is None:
            stop = self.arr.shape[axis_position]
        
        for index in range(start, stop):
            result.append(self.slice(axis, index))

        return(result)
    
    def expand_dim(self, axis):

        self.arr = np.expand_dims(self.arr, len(self.axes))
        self.axes.append(axis)
        self.shape_dict[axis] = 1
        self.shape = self.arr.shape