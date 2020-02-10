import torch
import numpy as np
from skimage import io
import cv2
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
import random

# training images: ./data/images/training/
# training gt: ./data/gt-images/training/

class ColonCellDataset(Dataset):
    def __init__(self, file_names, base_transform=None,transform=None, mask_transform=None, mode='train'):
        self.file_names = file_names
        self.base_transform=base_transform
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)
    
    def __getitem__(self, idx):
        'Generates one pair of image and mask'
        img_file_path = self.file_names[idx]
        image = load_image(img_file_path)
        mask = load_mask(img_file_path)

        #image = self.base_transform(image)
        #mask = self.base_transform(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() > 0.5:
            if random.random() > 0.5:
                image = TF.affine(image, 90, [0, 0], 1, 0)
                mask = TF.affine(mask, 90, [0, 0], 1, 0)
            else:
                image = TF.affine(image, -90, [0, 0], 1, 0)
                mask = TF.affine(mask, -90, [0, 0], 1, 0)

        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask

class SynthesizedDataset(Dataset):
    def __init__(self, file_names, base_transform=None,transform=None, mask_transform=None, mode='train'):
        self.file_names = file_names
        self.base_transform=base_transform
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)
    
    def __getitem__(self, idx):
        'Generates one pair of image and mask'
        img_file_path = self.file_names[idx]
        image = load_image(img_file_path)
        mask = load_mask_cropped(img_file_path)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() > 0.5:
            if random.random() > 0.5:
                image = TF.affine(image, 90, [0, 0], 1, 0)
                mask = TF.affine(mask, 90, [0, 0], 1, 0)
            else:
                image = TF.affine(image, -90, [0, 0], 1, 0)
                mask = TF.affine(mask, -90, [0, 0], 1, 0)

        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask
    
class ColonCellDatasetDouble(Dataset):
    def __init__(self, file_names, base_transform=None,transform=None, mask_transform=None, mode='train'):
        self.file_names = file_names
        self.base_transform=base_transform
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)
    
    def __getitem__(self, idx):
        'Generates one pair of image and mask'
        img_file_path = self.file_names[idx]
        image1 = load_image(img_file_path)
        image2 = load_image2(img_file_path)
        mask = load_mask(img_file_path)

        # Random horizontal flipping
        if random.random() > 0.5:
            image1 = TF.hflip(image1)
            image2 = TF.hflip(image2)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image1 = TF.vflip(image1)
            image2 = TF.vflip(image2)
            mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() > 0.5:
            if random.random() > 0.5:
                image1 = TF.affine(image1, 90, [0, 0], 1, 0)
                image2 = TF.affine(image2, 90, [0, 0], 1, 0)
                mask = TF.affine(mask, 90, [0, 0], 1, 0)
            else:
                image1 = TF.affine(image1, -90, [0, 0], 1, 0)
                image2 = TF.affine(image2, -90, [0, 0], 1, 0)
                mask = TF.affine(mask, -90, [0, 0], 1, 0)

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        mask = self.mask_transform(mask)

        return (image1, image2), mask


def load_image(path):
    img = Image.open(str(path))
    return img

def load_image2(path):
    #img = Image.open(str(path).replace('actin','DNA', 1))
    img = Image.open(str(path).replace('c001','c004', 1))
    return img

def load_mask(path):
    #mask = Image.open(str(path).replace('images', 'gt-images', 1).replace('actin.DIB', 'cells', 1))
    mask = Image.open(str(path).replace('marchantia_imgs', 'marchantia_gt', 1))
    return mask

def load_mask_cropped(path):
    #mask = Image.open(str(path).replace('cropped_imgs', 'cropped_gt', 1).replace('synth', 'gt', 1))
    mask = Image.open(str(path).replace('cropped_imgs', 'inverted_gt', 1).replace('synth', 'gt', 1))
    return mask