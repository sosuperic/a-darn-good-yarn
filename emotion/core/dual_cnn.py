# Network that has parallel paths for color image and black and white image


import tensorflow as tf

class DualCNN(object):
    # def __init__(self, params):
    def __init__(self, batch_size=None, img_w=None, img_h=None, output_dim=None, imgs=None, dropout_keep=None):
