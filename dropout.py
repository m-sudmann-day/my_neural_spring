import tensorflow as tf

class dropout():

    ratio = None
    keep_ratio = None

    def __init__(self, ratio):

        self.ratio = ratio
        self.keep_ratio = 1.0 - self.ratio

    def units_kept(self, total_units):

        return int(self.keep_ratio * total_units)

    def create(self, input_tensor):

        if self.keep_ratio < 1.0:
            return tf.nn.dropout(input_tensor, self.keep_ratio)
        else:
            return input_tensor