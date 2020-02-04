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

#######################

# This script will recursively segment a given image by reconsidering each class that the model outputs 
# and then inputting an image with only those pixels being left (the rest blacked out) and segment again.
# Note: this algorithm can take a very long time - might be worth using a smaller or downscaled image.

instance_model_filepath = './insseg_results/final_model.pt'

# Image to be recursively segmented
image_file_path = '../data/g1_t036_c001.png'

# Do you want to save images during the recursive progress? Note: will create potentially thousands of images
save_images = True

# Location to save images that document the recursive progress
recursive_results_location = './recursive_results'

# Where to save first segmentation iteration mask:
first_segmentation_results = './recursive_results/first_seg.npy'

color_lookup = np.load('../color_index_palette.npy')

#######################

global_instance_count_ = 0

def recursive_segmentation(original_image, input_image, model, transform, color_lookup_table):

    global global_instance_count_
    print(global_instance_count_)

    if save_images:
        norm = plt.Normalize(vmin=input_image.min(), vmax=input_image.max())
        saveimage = norm(input_image)
        plt.imsave(os.path.join(recursive_results_location, '{}newinput.png'.format(time.strftime("%m%d%Y-%H%M%S"))), saveimage)
    
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
    
    if save_images:
        norm = plt.Normalize(vmin=instance_masks[0].min(), vmax=instance_masks[0].max())
        saveimage = norm(instance_masks[0])
        plt.imsave(os.path.join(recursive_results_location, '{}newseg.png'.format(time.strftime("%m%d%Y-%H%M%S"))), saveimage)
    
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

model = UNet16(16, pretrained=True)
device = torch.device("cuda:0")
model = model.to(device)

# loading model
saved_state = torch.load(instance_model_filepath, map_location=device)
model.load_state_dict(saved_state)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

# input image to model
origimg = Image.open(image_file_path)

origimg = origimg.convert("RGB")
origimg = np.asarray(origimg, dtype=np.float32) / 255
origimg = origimg[:, :, :3]

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

np.save(first_segmentation_results, instance_masks[0])

norm = plt.Normalize(vmin=instance_masks[0].min(), vmax=instance_masks[0].max())
saveimage = norm(instance_masks[0])
plt.imsave(first_segmentation_results.replace('.npy','.png', 1), saveimage)

instance_masks= instance_masks[0]

instance_mask_index = instance_index_masks[0]

num_instances = len(np.unique(instance_mask_index))

for instance_index in range(1, num_instances):
    class_mask = np.where(instance_mask_index != instance_index, 0, instance_mask_index)

    class_mask_expanded = np.tile(np.expand_dims(class_mask,axis=2), (1,1,3))
    new_input_image = np.where(class_mask_expanded != 0, origimg, np.array([0,0,0], dtype = np.float32))

    recursive_segmentation(origimg, new_input_image, model, transform, color_lookup)

print("total number of instances: " + str(global_instance_count_))
