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

from dataset import ColonCellDataset, ColonCellDatasetDouble, SynthesizedDataset
from models import UNet11, UNet16, UNetResNet34, UNet11Double

import tqdm

class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = torch.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss

def validation_binary(model, criterion, valid_loader, num_classes=None):
    with torch.no_grad():
        model.eval()
        losses = []

        jaccard = []

        for inputs, targets in valid_loader:
            inputs = inputs.cuda(async=True)
            targets = targets.cuda(async=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            jaccard += get_jaccard(targets, (outputs > 0).float())
            #jaccard += get_jaccard(targets, outputs)

        valid_loss = np.mean(losses)  # type: float

        valid_jaccard = np.mean(jaccard).astype(np.float64)

        print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
        metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
        return metrics

def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + smooth))

def dice_loss_edited(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((3. * intersection + smooth) /
              ((iflat*iflat).sum() + (tflat*tflat).sum() + (tflat*tflat).sum() + smooth))

def intersection_over_union_loss(input,target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    iou = (intersection + 1) / ((iflat*iflat).sum() + (tflat*tflat).sum() - intersection + smooth)

    return 1 - iou

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

def train_model(model, criterion, train_loader, valid_loader, validation, optimizer, lr, num_epochs=20, batch_size=7):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    running_loss = 1.0

    loss_track = np.zeros([num_epochs])
    
    step = 0
    valid_losses = []
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
        
        tq = tqdm.tqdm(total=len(train_loader) * batch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []

        mean_loss = 0
        for i_batch, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(async=True)
            #inputs = (inputs[0].cuda(async=True), inputs[1].cuda(async=True))
            with torch.no_grad():
                targets = targets.cuda(async=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            batch_size = inputs.size(0)
            #batch_size = inputs[0].size(0)
            loss.backward()
            optimizer.step()
            step += 1
            tq.update(batch_size)
            losses.append(loss.item())
        
        tq.close()
        print('train loss', np.mean(losses))
        loss_track[epoch] = np.mean(losses)
        # valid_metrics = validation(model, criterion, valid_loader, num_classes=2)
        # valid_loss = valid_metrics['valid_loss']
        # valid_losses.append(valid_loss)

        if np.mean(losses) < running_loss:
            running_loss = np.mean(losses)
            best_model_wts = copy.deepcopy(model.state_dict())

    np.save('./results/inverted/vgg16_100_iou_pretrained_syn_2hand.npy', loss_track)
    print(loss_track)

    model.load_state_dict(best_model_wts)
    return model


data_dir = "./data"
batch_size = 25
num_epochs = 100

learning_rate = 0.001
momentum = 0.9

jaccard_weight = 0.5

device = torch.device("cuda:0")

input_size = 224 # VGG expects 224 RGB inputs
#model = UNet11(pretrained=True)
#model = UNet11Double(pretrained=True)
model = UNet16(pretrained=True)
#model = UNetResNet34(pretrained=True, is_deconv = True)

# training previous model
#model.load_state_dict(torch.load('./results/vgg16_colon1/model.pt'))
#model.load_state_dict(torch.load('./results/vgg11double/model.pt'))

model = model.to(device)

# Data transforms
base_transforms = transforms.Compose([
    #transforms.Resize(input_size),
    #transforms.RandomRotation(45, expand=False),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mask_data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ColonCellDataset(
    #glob(os.path.join(data_dir, 'images', 'training-2', '*-actin.DIB.png')),
    glob(os.path.join(data_dir, 'marchantia_imgs', '*1.png')),
    base_transform=base_transforms,
    transform=data_transforms,
    mask_transform=mask_data_transforms
    )

train_dataset_synth = SynthesizedDataset(
    #glob(os.path.join(data_dir, 'images', 'training-2', '*-actin.DIB.png')),
    #glob(os.path.join(data_dir, 'marchantia_imgs', '*.png')),
    glob(os.path.join(data_dir, 'cropped_imgs', '*.png')),
    base_transform=base_transforms,
    transform=data_transforms,
    mask_transform=mask_data_transforms
    )

val_dataset = ColonCellDataset(
    glob(os.path.join(data_dir, 'images', 'testing-2', '*-actin.DIB.png')),
    #glob(os.path.join(data_dir, 'marchantia_imgs', '*.png')),
    base_transform=base_transforms,
    transform=data_transforms,
    mask_transform=mask_data_transforms
    )

train_dataset2 = ColonCellDatasetDouble(
    #glob(os.path.join(data_dir, 'images', 'training-2', '*-actin.DIB.png')),
    glob(os.path.join(data_dir, 'marchantia_imgs', '*1.png')),
    base_transform=base_transforms,
    transform=data_transforms,
    mask_transform=mask_data_transforms
    )
val_dataset2 = ColonCellDatasetDouble(
    #glob(os.path.join(data_dir, 'images', 'testing-2', '*-actin.DIB.png')),
    glob(os.path.join(data_dir, 'marchantia_imgs', '*1.png')),
    base_transform=base_transforms,
    transform=data_transforms,
    mask_transform=mask_data_transforms
    )
    
#train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
train_loader = DataLoader(train_dataset_synth, shuffle=True, batch_size=batch_size, num_workers=0)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

train_loader_whole = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=0)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

gamma = 2

#criterion = LossBinary(jaccard_weight=jaccard_weight)
#criterion = nn.BCEWithLogitsLoss()
#criterion = FocalLoss(gamma)
#criterion = dice_loss
criterion = intersection_over_union_loss
validation = validation_binary

trained_model = train_model(
    model,
    criterion,
    train_loader,
    val_loader,
    validation,
    optimizer,
    lr=learning_rate,
    num_epochs=num_epochs,
    batch_size=batch_size
)

# trained_model = train_model(
#     trained_model,
#     criterion,
#     train_loader_whole,
#     val_loader,
#     validation,
#     optimizer,
#     lr=learning_rate,
#     num_epochs=10,
#     batch_size=1
# )

output_path = './results/inverted'

torch.save(trained_model.state_dict(), os.path.join(output_path, 'vgg16_100_iou_pretrained_syn_2hand.pt'))

trained_model.eval()

# with torch.no_grad():
#    for i, (inputs, targets) in enumerate(val_loader):
#        inputs = (inputs[0].to(device), inputs[1].to(device))
#        targets = targets.to(device)
#        outputs = trained_model(inputs)
#        outputs = outputs > 0.5
#        for j in range(outputs.size()[0]):
#            torchvision.utils.save_image(outputs[j], os.path.join(output_path, '%d_output.png' % (j)))
#            torchvision.utils.save_image(inputs[0][j], os.path.join(output_path, '%d_input1.png' % (j)))
#            torchvision.utils.save_image(inputs[1][j], os.path.join(output_path, '%d_input2.png' % (j)))
#            torchvision.utils.save_image(targets[j], os.path.join(output_path, '%d_gt.png' % (j)))
