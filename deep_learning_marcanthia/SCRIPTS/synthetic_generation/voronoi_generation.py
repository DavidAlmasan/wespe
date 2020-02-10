import cv2
import numpy as np 
import scipy.spatial as spatial
import os

from PIL import Image

from voronoi_fn import voronoi_finite_polygons_2d

########################################
# This script generates Voronoi diagrams that serve as ground truth data for the synthetic dataset. 
# It also generates the content images to be input into the neural style transfer script by varying the grayscale value of the diagram.

## Please change the following parameters to your pleasing!

# setting a random seed for the generation of random points
seed = 1234
np.random.seed(seed)

# Size of image to generate
img_height = 200
img_width = 200

# Number of cells in each image
number_of_cells = 100

# The number of images to be generated
number_images = 400

# Filepath for the images to be saved, relative to the location of this file.
# The ground truth images will be labelled ###_sem_gt.png, and the grayscale content images will be labelled ###_gray.png.
# The inverted ground truth image (with inner cells labelled) will also be saved as ###_invert_sem_gt.png
save_image_path = '../data/domB'

########################################

for i in range(number_images):

    gt_img = np.zeros((img_height,img_width,3), np.uint8)
    gray_img = np.zeros((img_height,img_width,3), np.uint8)

    x_list = np.random.randint(0, img_height, number_of_cells)
    y_list = np.random.randint(0, img_height, number_of_cells)

    vor = spatial.Voronoi(np.c_[x_list, y_list])

    points = vor.vertices

    # Obtaining the Voronoi polygons
    regions, vertices = voronoi_finite_polygons_2d(vor, i)

    for region in regions:
        cell_pts = vertices[region].reshape(-1,1,2)
        img_pts = np.round(cell_pts.astype(int))
        rand_brightness = np.random.randint(0, 255)
        cv2.polylines(gray_img, [img_pts], True, (rand_brightness, rand_brightness, rand_brightness), thickness=1)
        cv2.polylines(gt_img, [img_pts], True, (255, 255, 255), thickness=1)

    # Writing the images to the specified path
    cv2.imwrite(os.path.join(save_image_path, '{:03d}_sem_gt.png'.format(i)), gt_img)
    cv2.imwrite(os.path.join(save_image_path, '{:03d}_gray.png'.format(i)), gray_img)
    cv2.imwrite(os.path.join(save_image_path, '{:03d}_invert_sem_gt.png'.format(i)), (255 - gt_img))
