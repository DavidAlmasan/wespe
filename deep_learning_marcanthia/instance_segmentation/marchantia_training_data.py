from PIL import Image, ImageDraw
import numpy as np
import random

import glob

# img = Image.open('../colon_seg/data/marchantia_gt/invert_g2_t002_c001.png')
# img = img.convert("RGB")
# img_pixels = img.load()

# (imwidth, imheight) = img.size

# ## code to create color palette array of 4700 colors
# # num_colors = 4700
# # color_list = np.zeros((num_colors,3))
# # random.seed(100)
# # for i in range(1,num_colors):
# #     color_list[i] = (random.randint(1,254), random.randint(1,254), random.randint(1,254))
# # color_list = color_list.astype(int)
# # assert(np.unique(color_list,axis=0).shape[0] == num_colors)
# # np.save('./color_index_palette.npy', color_list)

# color_list = np.load('./color_index_palette.npy')

# # color_index 0 corresponds to black
# color_index = 1

# for i in range(imwidth):
#     for j in range(imheight):
#           if img_pixels[i,j] == (255, 255, 255):
#               ImageDraw.floodfill(img, (i, j), (color_list[color_index,0], color_list[color_index,1], color_list[color_index,2]))
#               color_index = color_index + 1

# print(len(img.getcolors(maxcolors=2000))-1)

# print(color_index-1)
# img.save('./marchantia_data/marchantia_gt/g2_t002_c001.png')

# print(np.array(img).shape)

################################################################################

# color_list = np.load('./color_index_palette.npy')
# color_index = 1

# content_list = []
# for filename in sorted(glob.glob('../colon_seg/data/inverted_gt/*_gt.png')):
#     im = Image.open(filename)
#     content_list.append(im)

# for (i, img) in enumerate(content_list):
#     color_index = 1
#     (imwidth, imheight) = img.size
#     img_pixels = img.load()
#     for ii in range(imwidth):
#         for jj in range(imheight):
#             if img_pixels[ii,jj] == (255, 255, 255):
#                 ImageDraw.floodfill(img, (ii, jj), (color_list[color_index,0], color_list[color_index,1], color_list[color_index,2]))
#                 color_index = color_index + 1
#     assert(len(img.getcolors(maxcolors=2000))-1 == color_index - 1)

#     img.save('./marchantia_data/all_cropped_gt/{:03d}_gt.png'.format(i))


################################################################################

color_list = np.load('./color_index_palette.npy')

file_names = glob.glob('./marchantia_data/cropped_ins_gt/*_cropped.png')

content_list = []
for filename in sorted(file_names):
    im = Image.open(filename)
    im_array = np.array(im)
    content_list.append(im_array)

for (i, img) in enumerate(content_list):
    img_flat = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    img_npy = np.zeros(img_flat.shape[0])

    img_file_name = sorted(file_names)[i]
    
    for row_i in range(img_flat.shape[0]):
        row = img_flat[row_i]
        if not np.all(row == [0,0,0]):
            img_npy[row_i] = np.where(np.all(color_list==row, axis=1))[0][0]

    img_npy = img_npy.reshape(img.shape[0],img.shape[1])
    np.save(str(img_file_name).replace('cropped_ins_gt','cropped_ins_npy_gt', 1).replace('.png', '.npy',1), img_npy)
