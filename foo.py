import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import PIL
import json
import tensorflow as tf
import datetime
try:import tensorflow_addons as tfa
except: pass
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add, LeakyReLU, Lambda, ReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, DepthwiseConv2D, GaussianNoise
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, DepthwiseConv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
import os
from scipy import ndimage, misc
import time
import re
from IPython import display

from utils import load_data, generate_and_save_images, load_dummy_data, img_to_patches, patches_to_img, remove_padding
from utils import gauss_kernel, _instance_norm
from loss_functions import content_loss, color_loss, texture_loss, content_loss_v2, tv_loss, total_loss_agg


import cv2
def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def model(imgShape):
    image = Input(imgShape)
    proc = Lambda(lambda x: x, output_shape=lambda x:x)(image) 
    # proc = gaussian_noise_layer(proc, 1)
    # Some noise to improve the training
    # proc = GaussianNoise(10)(proc)
    return Model(inputs=image, outputs=proc, name='a')

if __name__ == "__main__":
    a = load_data('./domA')
    a = (cv2.imread('./domA/g2_t017_c001 - Copy.png')[:, :, 0] - 127.5) / 127.5
    a = np.expand_dims(a, axis = 0)
    imgShape = a.shape
    a = np.expand_dims(a, axis = -1)
    print(a.shape)
    noise = tf.random.normal(a.shape, stddev = 0.05)
    img = a + noise
    b = model(imgShape)(img).numpy()
    print(np.unique(b))
    print(np.unique(noise))
    cv2.imshow('b', b[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()