import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

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

dist_matrix = np.load('./distance_matrix.npy')
color_lookup = np.load('./color_index_palette.npy')
instance_masks = np.load('./recursive_images_2/original_instance_mask_g1_t036.npy')

k = 2
index_to_consider = 120

idx = np.argpartition(dist_matrix[index_to_consider], k)

nearest_neighbours = idx[0:k+1]

instance_mask_index = color_to_index_image(instance_masks, color_lookup)

final_mask = np.zeros(instance_mask_index.shape)

for index in nearest_neighbours:
    if index != index_to_consider:
        class_mask = np.where(instance_mask_index != index, 0, instance_mask_index)
        final_mask = final_mask + class_mask
    else:
        # class_mask = np.where(instance_mask_index != index, 0, -1)
        class_mask = np.where(instance_mask_index != index, 0, instance_mask_index)
        final_mask = final_mask + class_mask

final_img = Image.new('RGB', final_mask.shape)
draw = ImageDraw.Draw(final_img)
final_img_orig = Image.new('RGB', final_mask.shape)
draw2 = ImageDraw.Draw(final_img_orig)

# original_img = Image.open('../marchantia_images/g1_t036_c001.png')
original_img = Image.open('../marchantia_images/g1_t036_with_nuclei.png')
original_img = original_img.convert('RGB')
original_img_array = np.asarray(original_img)
print(original_img_array.shape)

(imwidth, imheight) = final_mask.shape
for i in range(imwidth):
        for j in range(imheight):
                if int(final_mask[j][i]) != 0 and int(final_mask[j][i]) != -1:
                        draw.point((i,j), tuple(color_lookup[int(final_mask[j][i])]))
                        draw2.point((i,j), tuple(original_img_array[j][i]))                        
                elif int(final_mask[j][i]) == -1:
                        # draw.point((i,j), (255, 0, 0))
                        draw.point((i,j), tuple(color_lookup[int(final_mask[j][i])]))
                        draw2.point((i,j), tuple(original_img_array[j][i]))                        


# final_img.save('./nearest_neighbours_imgs/nearest_neighbors_test.png','PNG')
final_img_orig.save('./nearest_neighbours_imgs/original_image_test_120_2_nuclei.png','PNG')

