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
        
        # specifies location of the semantic ground truth image
        semantic_img = Image.open(str(img_file_path).replace('_synth', '_invert_sem_gt', 1))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = TF.to_grayscale(semantic_img, num_output_channels=1)
        
        # specifies location of the instance ground truth array
        instance_mask = np.load(str(img_file_path).replace('_synth', '_ins_gt', 1).replace('.png','.npy',1))

        # Random horizontal flipping
        if random.random() > 0.5:
            img = TF.hflip(img)
            semantic_img = TF.hflip(semantic_img)
            instance_mask = np.fliplr(instance_mask)

        # Random vertical flipping
        if random.random() > 0.5:
            img = TF.vflip(img)
            semantic_img = TF.vflip(semantic_img)
            instance_mask = np.flipud(instance_mask)

        img = img.convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255
        img = img[:, :, :3]
        
        instance_input = img # not blacking out the instance input

        img = self.transform(img)
        instance_input = self.transform(instance_input)
        semantic_mask = self.transform(semantic_img)

        instance_mask = np.ascontiguousarray(instance_mask)

        return img, semantic_mask, instance_mask, instance_input

    def __getitem__(self, index):
        'Generates one set of image, binary mask, and gt instance segmentation mask'
        
        image, semantic_mask, instance_mask, instance_input = self.__load_data(index)

        return image, semantic_mask, instance_mask, instance_input
