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
from torchsummary import summary
import matplotlib.pyplot as plt
import time
import os
import copy
from glob import glob

from dataset import CVPPPDataset, MarchantiaDataset
from loss import DiscriminativeLoss, dice_loss, intersection_over_union_loss
from models import UNet16

import tqdm

delta_v = 0.5
delta_d = 1.5

feature_dim = 16 # for CVPPP dataset as in paper
# feature_dim = 32 
# image_shape = (512, 512) # CVPPP
image_shape = (128, 128) # Marchantia

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
            # torch.save(model.state_dict(), os.path.join('./marchantia_results/insseg/16fdim_1000_session2', 'model_{}epochs_16fdim_unet16.pt'.format(epoch+1)))
            # np.save('./marchantia_results/insseg/16fdim_1000_session2/net16_{}epochs_16fdim.npy'.format(epoch+1), loss_track)
            torch.save(model.state_dict(), os.path.join('./marchantia_results/insseg//gray_variation_255', 'model_{}epochs_16fdim_unet16_255gray.pt'.format(epoch+1)))
            np.save('./marchantia_results/insseg/gray_variation_255/net16_{}epochs_16fdim_255gray.npy'.format(epoch+1), loss_track)

    # np.save('./marchantia_results/semseg/gray_variation/unet16_100epochs_255grayvar.npy', loss_track)
    # np.save('./marchantia_results/semseg/100epochs_16fdim_2/unet16_100epochs.npy', loss_track)
    # np.save('./marchantia_results/insseg/16fdim_1000_session2/unet16_1000epochs_16fdim.npy', loss_track)
    np.save('./marchantia_results/insseg/gray_variation_255/unet16_1000epochs_16fdim_gray255.npy', loss_track)

    model.load_state_dict(best_model_wts)
    return model

num_epochs = 1000
batch_size = 20

# data_dir = './CVPPP2017_LSC_training/training'
data_dir = './marchantia_data'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda:0')

sem_learning_rate = 0.001
momentum = 0.9

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

different_grayscale_synthetic_path = '../generation/images/synthesized_255_grayvar'

Marchantia_dataset = MarchantiaDataset(
    # glob(os.path.join(data_dir, 'cropped_imgs', '*.png')),
    glob(os.path.join(different_grayscale_synthetic_path, '*.png')), # for testing effect on synthetic data quality variation
    data_transforms
)

# CVPPP_dataset = CVPPPDataset(
#     glob(os.path.join(data_dir, 'A1', '*_rgb.png')), 
#     data_transforms
# )

Marchantia_loader = DataLoader(Marchantia_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
# CVPPP_loader = DataLoader(CVPPP_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

model = UNet16(feature_dim, pretrained=True)
model = model.to(device)

# seg_criterion = intersection_over_union_loss
seg_criterion = nn.BCEWithLogitsLoss()
ins_criterion = DiscriminativeLoss(delta_v, delta_d, feature_dim, image_shape)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=sem_learning_rate)

# output_path = './plant_results/semseg'
output_path = './marchantia_results/semseg'

#have_sem_model = input('Do you have a pretrained semantic segmentation model? (y if yes, anything else for no) ')
have_sem_model = 'y'

if have_sem_model == 'y':
    print('Loading pretrained model...')
    # model.load_state_dict(torch.load('./marchantia_results/semseg/model_100epochs_16fdim_unet16_bce.pt'))
    model.load_state_dict(torch.load('./marchantia_results/semseg/gray_variation/model_100epochs_16fdim_unet16_255grayvar.pt'))
    trained_seg_model = model
else:
    print('Training semantic segmentation network...')
    trained_seg_model = train_model(
        model,
        seg_criterion,
        # CVPPP_loader,
        Marchantia_loader,
        optimizer,
        num_epochs,
        batch_size,
        device,
        sem_seg = True
    )
    torch.save(trained_seg_model.state_dict(), os.path.join(output_path, '100epochs_16fdim_2/model_100epochs_16fdim_unet16_bce.pt'))
    # torch.save(trained_seg_model.state_dict(), os.path.join(output_path, 'gray_variation', 'model_100epochs_16fdim_unet16_255grayvar.pt'))

train_instance_network = True

if(train_instance_network):
    #summary(trained_seg_model, (3, 512, 512))
    #print(trained_seg_model)
    freeze_encoder = False
    if(freeze_encoder):
        child_counter = 0
        for child in trained_seg_model.children():
            if child_counter < 8:
                for param in child.parameters():
                    param.requires_grad = False
            child_counter += 1

    instance_optimizer = optim.Adam(filter(lambda p: p.requires_grad, trained_seg_model.parameters()), lr=sem_learning_rate)

    print('Training instance segmentation network...')
    trained_ins_model = train_model(
        trained_seg_model,
        ins_criterion,
        # CVPPP_loader,
        Marchantia_loader,
        instance_optimizer,
        num_epochs,
        batch_size,
        device,
        sem_seg = False
    )
    # torch.save(trained_ins_model.state_dict(), os.path.join('./plant_results/insseg', 'model_unet16_50epochs_frozenencoder.pt'))
    torch.save(trained_ins_model.state_dict(), os.path.join('./marchantia_results/insseg', 'model_1000epochs_16fdim_unet16.pt'))

