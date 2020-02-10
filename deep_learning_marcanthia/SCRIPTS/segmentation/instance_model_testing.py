import torch
import torch.nn as nn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from model_arch import UNet16
from clustering import get_instance_masks


import torchvision.transforms.functional as TF

##########################

# This script allows you to test your instance segmentation model with a test image.

# What input image do you want to test?
input_img_filepath = '../data/g1_t036_c001.png'

# Where is the model you want to test?
model_filepath = './insseg_results/final_model.pt'

# Where would you like to save the image?
save_img_location = './ins_test_result.png'

num_feature_dim = 16

##########################

model = UNet16(num_feature_dim, pretrained=True)

model.load_state_dict(torch.load(model_filepath))
model.eval()

device = torch.device("cuda:0")

model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()
])

with torch.no_grad():
    image = Image.open(input_img_filepath)

    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255
    image = image[:, :, :3]
    
    image = transform(image)
    image = image.unsqueeze(0)

    image = image.to(device)
    output = model(image)

    instance_output = output[1]
    instance_masks = get_instance_masks(instance_output)

norm = plt.Normalize(vmin=instance_masks[0][0].min(), vmax=instance_masks[0][0].max())

saveimage = norm(instance_masks[0][0])

# save the image
plt.imsave(save_img_location, saveimage)
    
