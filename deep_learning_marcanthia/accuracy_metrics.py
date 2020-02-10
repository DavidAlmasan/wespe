import numpy as np
from PIL import Image

def dice_coefficient(input, target):

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return ((2. * intersection) / ((iflat*iflat).sum() + (tflat*tflat).sum()))

def iou(input, target):

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return ((intersection) / ((iflat*iflat).sum() + (tflat*tflat).sum() - intersection))


output_path = './colon_seg/results/vgg16_colon1/5_output.png'
# output_path = './colon_seg/results/vgg16_colon1/marchantia2_after1marchantiaimage.png'
# gt_path = './colon_seg/data/marchantia_gt/g2_t002_c001.png'
gt_path = './colon_seg/results/vgg16_colon1/5_gt.png'

output_image = Image.open(output_path)
gt_image = Image.open(gt_path)
output_image = output_image.convert('L')
gt_image = gt_image.convert('L')

output_img_array = np.asarray(output_image)
gt_img_array = np.asarray(gt_image)

print('Image: ' + output_path)
print('Dice coefficient: ' + str(dice_coefficient(output_img_array, gt_img_array)))
print('IOU: ' + str(iou(output_img_array, gt_img_array)))


