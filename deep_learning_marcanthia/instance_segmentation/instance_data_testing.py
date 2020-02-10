from PIL import Image
import numpy as np 
import torch

color_list = np.load('./color_index_palette.npy')

img = Image.open('./test_instance1.png')
# img = Image.open('./CVPPP2017_LSC_training/training/A1/plant002_label.png')
print(img.mode)
mask = np.array(img)
print(mask.shape)
mask_flat = mask.reshape(mask.shape[0]*mask.shape[1], mask.shape[2])
print(mask_flat.shape)

final_mask = np.zeros(mask_flat.shape[0])
print(final_mask.shape)

for row_i in range(mask_flat.shape[0]):
    row = mask_flat[row_i]
    if not np.all(row == [0,0,0]):
        final_mask[row_i] = np.where(np.all(color_list==row, axis=1))[0][0]

final_mask = final_mask.reshape(mask.shape[0],mask.shape[1])
print(final_mask.shape)

np.save('saved_mask.npy', final_mask)

