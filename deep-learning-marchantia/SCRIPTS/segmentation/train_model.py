import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils as utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from glob import glob

from dataset import MarchantiaDataset
from losses import DiscriminativeLoss, dice_loss
from model_arch import UNet16

import tqdm

###################################

# This script trains a segmentation model. There are options for semantic segmentation and instance segmentation.
# The base architecture for the neural network is a UNet with a pre-trained VGG16 encoder.

#### Options/parameters:

# Are we training the semantic segmentation network? 
# If False, need to ensure we have a segmentation network already trained so that we can train the instance network
train_semantic_network = True

# Loss to train semantic segmentation. Recommended to use nn.BCEWithLogitsLoss()
seg_criterion = nn.BCEWithLogitsLoss() # can also train with dice_loss if desired

# Location of pre-trained semantic segmentation network or where to save semantic segmentation network after training:
semseg_network_filepath = './semseg_results/final_model.pt'

# Directory to save other semantic segmentation results.
semseg_results_filepath = './semseg_results'

# Are we training the instance segmentation network? It is recommended to train semantic and instance networks in separate sessions.
train_instance_network = False

# Freeze the encoder for instance segmentation training? It is recommended not to.
freeze_encoder = False

# Directory to save instance results. The network is saved automatically every 100 epochs as well as the loss data
insseg_results_filepath = './insseg_results'

# Size of images when training with instance segmentation
image_shape = (128, 128)

# Directory where training images are located. This set-up assumes the following files in this directory:
# xxx_synth.png -- the synthetic Marchantia images
# xxx_ins_gt.npy -- the instance mask ground truth arrays
# xxx_invert_sem_gt.png -- the semantic ground truth images with the inner cells labelled
# In order to add your own hand-labelled Marchantia images, please ensure that they are labelled in this way in this directory and cropped to the correct size
training_images_filepath = '../data/synthesized_imgs'

# Parameters for the discriminative loss function
delta_v = 0.5
delta_d = 1.5
feature_dim = 16 # feature dimension to embed pixels for the instance segmentation

# Other training parameters
learning_rate = 0.001
num_epochs = 1000 # number of epochs to run
batch_size = 20 # batch size

###################################

def train_model(model, criterion, train_loader, optimizer, num_epochs, batch_size, device, sem_seg=False):

    best_model_wts = copy.deepcopy(model.state_dict())
    running_loss = 100.

    loss_track = np.zeros([num_epochs])

    step = 0
    for epoch in range(num_epochs):

        model.train()

        tq = tqdm.tqdm(total=len(train_loader) * batch_size)
        tq.set_description('Epoch {}'.format(epoch))
        losses = []

        for i_batch, (inputs, sem_targets, ins_targets, ins_inputs) in enumerate(train_loader):
            
            if sem_seg:
                targets = sem_targets
                output_index = 0
            else:
                inputs = ins_inputs
                targets = ins_targets  
                output_index = 1

            inputs = inputs.cuda(device, async=True)
            with torch.no_grad():
                targets = targets.cuda(device, async=True)
            targets = targets.type(torch.cuda.FloatTensor)

            outputs = model(inputs)
            loss = criterion(outputs[output_index], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            
            tq.update(batch_size)
            losses.append(loss.item())
        
        tq.close()
        print('train loss', np.mean(losses))
        loss_track[epoch] = np.mean(losses)

        if np.mean(losses) < running_loss:
            running_loss = np.mean(losses)
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 100 == 0 and sem_seg == False:
            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), os.path.join(insseg_results_filepath, '{}epochs_model.pt'.format(epoch+1)))
            np.save(os.path.join(insseg_results_filepath, '{}epochs_losses.npy'.format(epoch+1)), loss_track)

    if sem_seg == False:
        np.save(os.path.join(insseg_results_filepath, 'final_losses.npy'), loss_track)
    else:
        np.save(os.path.join(semseg_results_filepath, 'final_losses.npy'), loss_track)

    model.load_state_dict(best_model_wts)
    return model

device = torch.device('cuda:0')

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

Marchantia_dataset = MarchantiaDataset(
    glob(os.path.join(training_images_filepath, '*_synth.png')),
    data_transforms
)

Marchantia_loader = DataLoader(Marchantia_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

model = UNet16(feature_dim, pretrained=True)
model = model.to(device)

ins_criterion = DiscriminativeLoss(delta_v, delta_d, feature_dim, image_shape)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if not train_semantic_network:
    print('Loading pretrained model...')
    model.load_state_dict(torch.load(semseg_network_filepath))
    trained_seg_model = model
else:
    print('Training semantic segmentation network...')
    trained_seg_model = train_model(
        model,
        seg_criterion,
        Marchantia_loader,
        optimizer,
        num_epochs,
        batch_size,
        device,
        sem_seg = True
    )
    torch.save(trained_seg_model.state_dict(), semseg_network_filepath)

if(train_instance_network):

    if(freeze_encoder):
        child_counter = 0
        for child in trained_seg_model.children():
            if child_counter < 8:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1

    instance_optimizer = optim.Adam(filter(lambda p: p.requires_grad, trained_seg_model.parameters()), lr=learning_rate)

    print('Training instance segmentation network...')
    trained_ins_model = train_model(
        trained_seg_model,
        ins_criterion,
        Marchantia_loader,
        instance_optimizer,
        num_epochs,
        batch_size,
        device,
        sem_seg = False
    )
    torch.save(trained_ins_model.state_dict(), os.path.join(insseg_results_filepath, 'final_model.pt'))

