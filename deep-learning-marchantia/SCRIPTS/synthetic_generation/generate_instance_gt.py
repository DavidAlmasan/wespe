from PIL import Image, ImageDraw
import numpy as np
import os

import glob

########################

# This script generates the ground truth data for instance segmentation, given the ground truth images for the 
# semantic segmentation (cells themselves labelled white, everything else is black).

# Locations of the semantic ground truth data - IMPORTANT: must be ground truth with inner cells labelled! NOT cell walls
file_names = glob.glob('../data/synthesized_imgs/*_invert_sem_gt.png')

# Directory to save instance ground truth data. 
# The instance image will be saved as ###_ins_gt.png and the colour array version as ###_ins_gt.npy.
instance_save_file_path = '../data/synthesized_imgs'

# Loading the color-lookup table
color_list = np.load('../color_index_palette.npy')

########################

content_list = []
for filename in sorted(file_names):
    im = Image.open(filename)
    content_list.append(im)

ins_img_list = []
for (i, img) in enumerate(content_list):
    color_index = 1
    (imwidth, imheight) = img.size
    img_pixels = img.load()
    for ii in range(imwidth):
        for jj in range(imheight):
            if img_pixels[ii,jj] == (255, 255, 255):
                ImageDraw.floodfill(img, (ii, jj), (color_list[color_index,0], color_list[color_index,1], color_list[color_index,2]))
                color_index = color_index + 1

    ins_img_list.append(np.array(img))
    img.save(os.path.join(instance_save_file_path, '{:03d}_ins_gt.png'.format(i)))

for (i, img) in enumerate(ins_img_list):
    img_flat = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    img_npy = np.zeros(img_flat.shape[0])
    
    for row_i in range(img_flat.shape[0]):
        row = img_flat[row_i]
        if not np.all(row == [0,0,0]):
            img_npy[row_i] = np.where(np.all(color_list==row, axis=1))[0][0]

    img_npy = img_npy.reshape(img.shape[0],img.shape[1])
    np.save(os.path.join(instance_save_file_path, '{:03d}_ins_gt.npy'.format(i)), img_npy)
