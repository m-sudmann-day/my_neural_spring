
class activation():
    
    image_x1 = None
    image_y1 = None
    image_x2 = None
    image_y2 = None
    tile = None

    def __init__(self, x, y, value, layer, image_num):
        self.x = x
        self.y = y
        self.value = value
        self.layer = layer
        self.image_num = image_num

    def calculate_image_coordinates(self):
        #print(self.layer.name, self.x, self.image_x1, self.image_x2)
        (self.image_x1, self.image_y1, self.image_x2, self.image_y2) = self.layer.calculate_image_coordinates(self.x, self.y, self.x, self.y)
        self.tile = self.layer.get_tile(self.x, self.y, self.x, self.y)

    def get_tile(self, first_image_num_in_batch):
        
        self.tile = self.layer.get_tile(self.image_num - first_image_num_in_batch, self.x, self.y, self.x, self.y)
