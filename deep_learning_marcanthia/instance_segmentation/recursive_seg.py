import torch
import torch.nn as nn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import copy
from PIL import Image
from models import UNet16
from clustering import get_instance_masks

import torchvision.transforms.functional as TF

# This script will recursively segment a given image by reconsidering each class that the model outputs 
# and then inputting an image with only those pixels being left (the rest blacked out) and segment again

global_instance_count_ = 0

def recursive_segmentation(original_image, input_image, model, transform, color_lookup_table):

    global global_instance_count_
    print(global_instance_count_)

    norm = plt.Normalize(vmin=input_image.min(), vmax=input_image.max())
    saveimage = norm(input_image)
    plt.imsave('./recursive_images_6_37/newinput{}.png'.format(time.strftime("%m%d%Y-%H%M%S")), saveimage)
    
    # stop segmenting if the number of nonzero pixels of input image is equivalent or smaller to a 8x8 block of pixels
    num_nonzero_pixels = np.count_nonzero(input_image) / 3
    if num_nonzero_pixels <= 64:
        global_instance_count_ += 1
        return

    with torch.no_grad():
        # input image is np array with RGB channels
        input_image = transform(input_image)
        input_image = input_image.unsqueeze(0)
        input_image = input_image.to(device)

        model_output = model(input_image)
        [instance_masks, instance_index_masks] = get_instance_masks(model_output[1])
    
    norm = plt.Normalize(vmin=instance_masks[0].min(), vmax=instance_masks[0].max())
    saveimage = norm(instance_masks[0])
    plt.imsave('./recursive_images_6_37/newseg{}.png'.format(time.strftime("%m%d%Y-%H%M%S")), saveimage)
    
    index_image = instance_index_masks[0]
    
    num_instances = len(np.unique(index_image))

    # stop segmenting if only one instance is counted (1 instance + black background = 2) 
    if num_instances <= 2:
        global_instance_count_ += 1
        return

    for instance_index in range(1, num_instances):
        # The mask of the relevant class only - the rest zero
        class_mask = np.where(index_image != instance_index, 0, index_image)
        class_mask_expanded = np.tile(np.expand_dims(class_mask,axis=2), (1,1,3))

        # this image is the original image with only the relevant parts left (the rest blacked out)
        new_input_image = np.where(class_mask_expanded != 0, original_image, np.array([0,0,0], dtype = np.float32))

        recursive_segmentation(original_image, new_input_image, model, transform, color_lookup_table)
    
    return

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 4700x3 matrix, each row s 
color_lookup = np.load('./color_index_palette.npy')

model = UNet16(16, pretrained=True)

device = torch.device("cuda:0")

model = model.to(device)

# loading model
saved_state = torch.load('./marchantia_results/insseg/16fdim_1000_session2/model_1000epochs_16fdim_unet16.pt', map_location=device)
model.load_state_dict(saved_state)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# image_file_path = '../marchantia_images/g1_t036_c001_downscale512.png'
image_file_path = '../marchantia_images/g6_t037_c001_downscale512.png'
# image_file_path = './g1_t036_c001.png'

have_first_instance_run = False

# input image to model
origimg = Image.open(image_file_path)

origimg = origimg.convert("RGB")
origimg = np.asarray(origimg, dtype=np.float32) / 255
origimg = origimg[:, :, :3]

if not have_first_instance_run:
    with torch.no_grad():
        image = transform(origimg)
        image = image.unsqueeze(0)

        image = image.to(device)
        output = model(image)
        
        # semantic output is output[0], instance output is output[1]
        instance_output = output[1]
        
        # instance_masks contains the color (not color index!) for each pixel in the image (e.g. for a 1024x1024 image, it is a 1024x1024 matrix)
        [instance_masks, instance_index_masks] = get_instance_masks(instance_output)
        print(instance_masks[0].shape)

    np.save('./recursive_images_6_37/original_instance_mask_g6_t037_downscale512.npy', instance_masks[0])
    instance_masks= instance_masks[0]

else:
    instance_masks = np.load('./recursive_images_6_37/original_instance_mask_g6_t037_downscale512.npy')

instance_mask_index = instance_index_masks[0]

num_instances = len(np.unique(instance_mask_index))
# num_instances = 2

for instance_index in range(1, num_instances):
    class_mask = np.where(instance_mask_index != instance_index, 0, instance_mask_index)

    class_mask_expanded = np.tile(np.expand_dims(class_mask,axis=2), (1,1,3))
    new_input_image = np.where(class_mask_expanded != 0, origimg, np.array([0,0,0], dtype = np.float32))

    recursive_segmentation(origimg, new_input_image, model, transform, color_lookup)

    ################

    # class_image = np.where(class_mask_expanded != 0, origimg, np.array([0,0,0], dtype = np.float32))

    # norm2 = plt.Normalize(vmin=class_image.min(), vmax=class_image.max())
    # saveimage2 = norm2(class_image)
    # plt.imsave('test_recursive2.png', saveimage2)

print("total number of instances: " + str(global_instance_count_))
