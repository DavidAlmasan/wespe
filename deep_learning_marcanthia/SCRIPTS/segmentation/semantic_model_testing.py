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
from . import model_arch

import torchvision.transforms.functional as TF

def test_semantic_model(inputImgPath = '../data/g1_t036_c001.png', modelPath = '../semseg_model_100epochs_16fdim_unet16_bce.pt', saveImgPath = './sem_test_result_NEW.png', verbose = 1):
    curFolder = os.path.abspath(os.path.dirname(__file__))
    inputImgPath = os.path.join(curFolder, inputImgPath)
    modelPath = os.path.join(curFolder, modelPath)
    saveImgPath = os.path.join(curFolder, saveImgPath)
    num_feature_dim = 16
    model = model_arch.UNet16(num_feature_dim, pretrained=True)

    model.load_state_dict(torch.load(modelPath))
    model.eval()

    device = torch.device("cuda:0")

    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with torch.no_grad():
        image = Image.open(inputImgPath)
        if verbose:
            print('Image at path {} has shape {}.'.format(inputImgPath, np.asarray(image).shape))
        image = image.convert("RGB")
        print('Image converted to RGB has shape: ', np.asarray(image).shape)
        image = np.asarray(image, dtype=np.float32) / 255
        image = image[:, :, :3]
        
        image = transform(image)
        image = image.unsqueeze(0)

        image = image.to(device)
        output = model(image)

        output_img = output[0]
        torchvision.utils.save_image(output_img, saveImgPath)
        
if __name__ == "__main__":
    ##########################

    # This script allows you to test your semantic segmentation model with a test image.

    # What input image do you want to test?
    input_img_filepath = '../data/g1_t036_c001.png'

    # Where is the model you want to test?
    model_filepath = '../semseg_model_100epochs_16fdim_unet16_bce.pt'

    # Where would you like to save the image?
    save_img_location = './sem_test_result.png'
    test_semantic_model(input_img_filepath, model_filepath, save_img_location)
