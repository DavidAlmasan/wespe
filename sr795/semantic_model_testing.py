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

import torchvision.transforms.functional as TF

##########################

# This script allows you to test your semantic segmentation model with a test image.

# What input image do you want to test?
input_img_filepath = '../test_data/200224_LQ_LQ-Position001.tiff'
input_img_filepath = '../watershed_comparison_photos/original.tiff'
input_img_filepath = '../watershed_comparison_photos/enhanced.png'
#input_img_filepath = '../data/test2.tiff'
#input_img_filepath = '../data/marcanthia_enhanced.png' 
#input_img_filepath = '../data/trial.png'
# Where is the model you want to test?
#model_filepath = './semseg_results/check_point_clean_seg.pt'
#model_filepath = '../data/semseg_model_100epochs_16fdim_unet16_bce.pt'
model_filepath = '../../../../sr795/DeepLearningFourthYear/deep-learning-marchantia/SCRIPTS/segmentation/semseg_results/final_model.pt'
# Where would you like to save the image?
save_img_location = './sem_enhanced.png'

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
    #model = model.train()
    image = Image.open(input_img_filepath)

    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255
    image = image[:, :, :3]
    
    image = transform(image)
    image = image.unsqueeze(0)

    image = image.to(device)
    results = np.zeros([1,1,1024,1024])
    for i in range(0):

        output = model(image)
    
        output_img = torch.sigmoid(output[0])
        result1 = output_img.data.cpu().numpy()
        results = np.concatenate((results,result1))
        print(output_img.shape)
        print(output_img[0,0,500:502,500:502])

    model = model.eval()
    output = model(image)
    
    output_img = torch.sigmoid(output[0])
    print(output_img[0,0,500:502,500:502])
    torchvision.utils.save_image(output_img, save_img_location)
    np.save("./file.npy",results)
