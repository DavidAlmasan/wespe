import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms

import numpy as np

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        
        self.pool = nn.MaxPool2d(2,2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0], self.relu)
        self.conv2 = nn.Sequential(self.encoder[3], self.relu)
        self.conv3 = nn.Sequential(self.encoder[6], self.relu,
                                    self.encoder[8], self.relu)
        self.conv4 = nn.Sequential(self.encoder[11], self.relu, 
                                    self.encoder[13], self.relu)
        self.conv5 = nn.Sequential(self.encoder[16], self.relu,
                                    self.encoder[18], self.relu)

        # DecoderBlock(in_channels, middle_channels, out_channels)
        self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=True)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        
        return x_out
    
class UNet11Double(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        
        self.pool = nn.MaxPool2d(2,2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0], self.relu)
        self.conv2 = nn.Sequential(self.encoder[3], self.relu)
        self.conv3 = nn.Sequential(self.encoder[6], self.relu,
                                    self.encoder[8], self.relu)
        self.conv4 = nn.Sequential(self.encoder[11], self.relu, 
                                    self.encoder[13], self.relu)
        self.conv5 = nn.Sequential(self.encoder[16], self.relu,
                                    self.encoder[18], self.relu)

        # DecoderBlock(in_channels, middle_channels, out_channels)
        self.center = DecoderBlock((256 + num_filters * 8)*2, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec5 = DecoderBlock(512*2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512*2 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=True)
        self.dec3 = DecoderBlock(256*2 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128*2 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64*2 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        conv1_1 = self.conv1(x[0])
        conv2_1 = self.conv2(self.pool(conv1_1))
        conv3_1 = self.conv3(self.pool(conv2_1))
        conv4_1 = self.conv4(self.pool(conv3_1))
        conv5_1 = self.conv5(self.pool(conv4_1))
        
        conv1_2 = self.conv1(x[1])
        conv2_2 = self.conv2(self.pool(conv1_2))
        conv3_2 = self.conv3(self.pool(conv2_2))
        conv4_2 = self.conv4(self.pool(conv3_2))
        conv5_2 = self.conv5(self.pool(conv4_2))       
        
        center = self.center(self.pool(torch.cat([conv5_1, conv5_2], 1)))

        dec5 = self.dec5(torch.cat([center, conv5_1, conv5_2], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4_1, conv4_2], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3_1, conv3_2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2_1, conv2_2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1_1, conv1_2], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        
        return x_out

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)        

        self.num_classes = num_classes

        self.encoder = models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0], self.relu,
                                   self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(self.encoder[5], self.relu,
                                   self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(self.encoder[10], self.relu,
                                   self.encoder[12], self.relu,
                                   self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(self.encoder[17], self.relu,
                                   self.encoder[19], self.relu,
                                   self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(self.encoder[24], self.relu,
                                   self.encoder[26], self.relu,
                                   self.encoder[28], self.relu)

        # DecoderBlock(in_channels, middle_channels, out_channels)
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out
    
class UNet19(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2) 
        
        self.num_classes = num_classes

        self.encoder = models.vgg19(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0], self.relu,
                                   self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(self.encoder[5], self.relu,
                                   self.encoder[7], self.relu)




        # DecoderBlock(in_channels, middle_channels, out_channels)
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=True)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv=True)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=True)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        def forward(self, x):
            conv1 = self.conv1(x)
            conv2 = self.conv2(self.pool(conv1))
            conv3 = self.conv3(self.pool(conv2))
            conv4 = self.conv4(self.pool(conv3))
            conv5 = self.conv5(self.pool(conv4))
            center = self.center(self.pool(conv5))

            dec5 = self.dec5(torch.cat([center, conv5], 1))
            dec4 = self.dec4(torch.cat([dec5, conv4], 1))
            dec3 = self.dec3(torch.cat([dec4, conv3], 1))
            dec2 = self.dec2(torch.cat([dec3, conv2], 1))
            dec1 = self.dec1(torch.cat([dec2, conv1], 1))


            if self.num_classes > 1:
                x_out = F.log_softmax(self.final(dec1), dim=1)
            else:
                x_out = self.final(dec1)
            
            return x_out

    
class UNetResNet34(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out
