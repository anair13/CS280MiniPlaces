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
PAD = 0 # the outermost pixels are seen less frequently by the receptive fields, so it might help to exclude them
STEP = (SPACE - PAD - PAD) / Q

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

            for x in range(PAD, SPACE+1-PAD, STEP):
                for y in range(PAD, SPACE+1-PAD, STEP):
                    top[0].data[j, :, :, :] = crop(img_before, x, y)
                    j = j + 1

    def backward(self, top, propagate_down, bottom):
        pass