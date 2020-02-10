import numpy as np
from scipy.spatial import distance_matrix
from scipy import ndimage
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import glob

# 1) get the centroids for each instance by going through each instance and finding all the corresponding pixels, and then averaging
# 2) build a distance matrix from centroid locations

def get_instance_centroids(instance_matrix):
    num_instances = len(np.unique(instance_matrix))
    print(num_instances)

    centroids = []

    for instance in range(num_instances):
        class_mask = np.where(instance_matrix != instance, 0, 1)

        centroid_value = ndimage.measurements.center_of_mass(class_mask)
        centroid_value1 = (int(round(centroid_value[1])), int(round(centroid_value[0]))) # saving in (x,y) pixel format for drawing
        # centroid_value2 = (int(round(centroid_value[1]))+1, int(round(centroid_value[0]))) # saving in (x,y) pixel format for drawing
        # centroid_value3 = (int(round(centroid_value[1])), int(round(centroid_value[0]))+1) # saving in (x,y) pixel format for drawing
        # centroid_value4 = (int(round(centroid_value[1])+1), int(round(centroid_value[0]))+1) # saving in (x,y) pixel format for drawing

        centroids.append(centroid_value1)
        # centroids.append(centroid_value2)
        # centroids.append(centroid_value3)
        # centroids.append(centroid_value4)

    return centroids

def color_to_index_image(color_image, color_lookup_table):
    color_image_flat = color_image.reshape(color_image.shape[0]*color_image.shape[1], color_image.shape[2])

    index_image = np.zeros(color_image_flat.shape[0])
    for row_i in range(color_image_flat.shape[0]):
        row = color_image_flat[row_i]
        if not np.all(row == [0,0,0]):
                if (np.where(np.all(color_lookup_table==row, axis=1)))[0].size != 0:
                        index_image[row_i] = np.where(np.all(color_lookup_table==row, axis=1))[0][0]
    
    index_image = index_image.reshape(color_image.shape[0], color_image.shape[1])

    return index_image

###############

color_lookup = np.load('./color_index_palette.npy')

instance_masks = np.load('./recursive_images_2/original_instance_mask_g1_t036.npy')
# instance_masks = np.load('./recursive_images_3/original_instance_mask_g1_t036_downscale512.npy')

# instance_img = Image.open('marchantia_results/insseg/16fdim_1000_session2/result6_37_1000epochs_16fdim.png')
# instance_img = instance_img.convert("RGB")
# instance_masks = np.asarray(instance_img)
# print(instance_masks.shape)

instance_mask_index = color_to_index_image(instance_masks, color_lookup)

centroids = get_instance_centroids(instance_mask_index)

img_shape = instance_mask_index.shape

# Drawing centroid image
# centroid_img = Image.new('L', img_shape)
# draw = ImageDraw.Draw(centroid_img)
# draw.point(centroids, fill=255)
# draw.point(centroids, fill=255)

# centroid_img.save('./centroid_images/g1_t036_centroids_2.png', 'PNG')

centroids_array = np.array(centroids, dtype='int')
dist_matrix = distance_matrix(centroids_array, centroids_array)

print(dist_matrix.shape)
np.save('./distance_matrix.npy', dist_matrix)

# plt.imshow(dist_matrix, interpolation='nearest', cmap=cm.Greys_r)

# plt.savefig('./centroid_images/g6_t037_distance_matrix.png', dpi = 150)

#################

# centroids = []

# for filename in glob.glob('./recursive_images_4/newseg*.png'):
#         img = Image.open(filename)
#         img = img.convert("RGB")
#         img_array = np.array(img)
#         img_array_flat = img_array.reshape(img_array.shape[0]*img_array.shape[1], img_array.shape[2])
#         num_instances = len(np.unique(img_array_flat, axis=0))
#         img_shape = img_array.shape

#         # if there is only one instance in the image, calculate the centroid and add it to the list of centroids
#         if num_instances == 2:
#                 centroid_value = ndimage.measurements.center_of_mass(img_array)
#                 centroid_value = (int(round(centroid_value[1])), int(round(centroid_value[0])))
#                 centroids.append(centroid_value)
                

# # Drawing centroid image
# centroid_img = Image.new('L', (img_shape[0], img_shape[1]))
# draw = ImageDraw.Draw(centroid_img)
# draw.point(centroids, fill=255)

# centroid_img.save('./test_centroids_recursive4.png', 'PNG')

# centroids_array = np.array(centroids_list, dtype='int')
# dist_matrix = distance_matrix(centroids_array, centroids_array)


