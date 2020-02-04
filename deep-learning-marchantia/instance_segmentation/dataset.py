import torch
import numpy as np
from skimage import io
import cv2
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
import random

class MarchantiaDataset(Dataset):
    def __init__(self, file_names, transform=None, mode='train'):
        self.file_names = file_names
        self.transform=transform
        self.mode = mode
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)
    
    def __load_data(self, index):

        img_file_path = self.file_names[index]

        img = Image.open(str(img_file_path))
        
        # semantic_img = Image.open(str(img_file_path).replace('cropped_imgs', 'cropped_sem_gt', 1).replace('synth', 'gt', 1))
        semantic_img = Image.open(str(img_file_path).replace('../generation/images/synthesized_255_grayvar', './marchantia_data/cropped_sem_gt', 1).replace('_synth', '_gt', 1))
        semantic_img = semantic_img.convert("RGB")
        # semantic_img_mask = np.asarray(semantic_img, dtype=np.float32) / 255
        semantic_img = TF.to_grayscale(semantic_img, num_output_channels=1)

        # instance_img = Image.open(str(img_file_path).replace('cropped_imgs', 'cropped_ins_npy_gt', 1).replace('synth', 'gt', 1))
        
        # instance_mask = np.load(str(img_file_path).replace('cropped_imgs', 'cropped_ins_npy_gt', 1).replace('synth', 'gt', 1).replace('.png','.npy',1))
        instance_mask = np.load(str(img_file_path).replace('../generation/images/synthesized_255_grayvar', './marchantia_data/cropped_ins_npy_gt', 1).replace('_synth', '_gt', 1).replace('.png', '.npy', 1))

        # Random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            semantic_img = TF.hflip(semantic_img)
            # instance_img = TF.hflip(instance_img)
            instance_mask = np.fliplr(instance_mask)

        # Random vertical flipping
        if random.random() > 0.5:
            img = TF.vflip(img)
            semantic_img = TF.vflip(semantic_img)
            # instance_img = TF.vflip(instance_img)
            instance_mask = np.flipud(instance_mask)

        img = img.convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255
        img = img[:, :, :3]
        
        # instance_input = np.where(semantic_img_mask == 0, np.array([0,0,0], dtype = np.float32), img)
        instance_input = img # not blacking out the instance input

        img = self.transform(img)
        instance_input = self.transform(instance_input)
        semantic_mask = self.transform(semantic_img)

        # instance_mask = np.array(instance_img)
        instance_mask = np.ascontiguousarray(instance_mask)

        return img, semantic_mask, instance_mask, instance_input

    def __getitem__(self, index):
        'Generates one set of image, binary mask, and gt instance segmentation mask'
        
        image, semantic_mask, instance_mask, instance_input = self.__load_data(index)

        return image, semantic_mask, instance_mask, instance_input

class CVPPPDataset(Dataset):
    def __init__(self, file_names, transform=None, mode='train'):
        self.file_names = file_names
        self.transform=transform
        self.mode = mode
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_names)
    
    def __load_data(self, index):

        img_file_path = self.file_names[index]

        img = Image.open(str(img_file_path))
        img = TF.resize(img, (512, 512))
        
        semantic_img = Image.open(str(img_file_path).replace('rgb', 'fg', 1))
        semantic_img = TF.resize(semantic_img, (512, 512))
        semantic_img = semantic_img.convert("RGB")
        semantic_img_mask = np.asarray(semantic_img, dtype=np.float32) / 255
        semantic_img = TF.to_grayscale(semantic_img, num_output_channels=1)

        instance_img = Image.open(str(img_file_path).replace('rgb', 'label', 1))
        instance_img = TF.resize(instance_img, (512, 512))

        # Random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            semantic_img = TF.hflip(semantic_img)
            instance_img = TF.hflip(instance_img)

        # Random vertical flipping
        if random.random() > 0.5:
            img = TF.vflip(img)
            semantic_img = TF.vflip(semantic_img)
            instance_img = TF.vflip(instance_img)

        img = img.convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255
        img = img[:, :, :3]
        
        instance_input = np.where(semantic_img_mask == 0, np.array([0,0,0], dtype = np.float32), img)

        img = self.transform(img)
        instance_input = self.transform(instance_input)
        semantic_mask = self.transform(semantic_img)

        instance_mask = np.array(instance_img)

        return img, semantic_mask, instance_mask, instance_input

    def __getitem__(self, index):
        'Generates one set of image, binary mask, and gt instance segmentation mask'
        
        image, semantic_mask, instance_mask, instance_input = self.__load_data(index)

        return image, semantic_mask, instance_mask, instance_input
