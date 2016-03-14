import caffe
import numpy as np
import random
import cv2
import math

IN = 128
OUT = 96
Q = 4 # number of "bins" in each dimension
OUTPUTS = (Q + 1) * (Q + 1)
SPACE = IN - OUT
STEP = SPACE / Q

def crop(img, x, y):
    new_img = img[:, y:y+OUT, x:x+OUT]
    return new_img

class WindowLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        pass

    def reshape(self, bottom, top):
        N = bottom[0].num # batch size
        top[0].reshape(N * OUTPUTS, 3, OUT, OUT)
    
    def forward(self, bottom, top):
        N = bottom[0].num # batch size
        j = 0 # indexes layers of top
        for i in range(N):
            img_before = bottom[0].data[i, :, :, :]

            for x in range(0, SPACE+1, STEP):
                for y in range(0, SPACE+1, STEP):
                    top[0].data[j, :, :, :] = crop(img_before, x, y)
                    j = j + 1

    def backward(self, top, propagate_down, bottom):
        pass