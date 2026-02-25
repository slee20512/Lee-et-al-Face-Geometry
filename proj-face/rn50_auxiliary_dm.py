import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

class rn50_auxiliary_dm(nn.Module):
    def __init__(self):
        super(rn50_auxiliary_dm, self).__init__()
        
        ## copy from rn50 
        ori_rn50 = models.__dict__['resnet50']()
        # encoder 
        self.conv1 = ori_rn50.conv1
        self.bn1 = ori_rn50.bn1
        self.relu = ori_rn50.relu
        self.maxpool = ori_rn50.maxpool
        self.layer1 = ori_rn50.layer1
        self.layer2 = ori_rn50.layer2
        self.layer3 = ori_rn50.layer3
        # identity network
        self.layer4 = ori_rn50.layer4
        self.avgpool = ori_rn50.avgpool
        self.fc = ori_rn50.fc
        # depth network
        self.depth_predictor = DepthEnDecoder(cin=1024, cout=1, nf=64, zdim=256)
    
    def forward(self, x):
        # unified encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # pred depth
        depth_pred = self.depth_predictor(x).squeeze(1)
        # pred identity
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, depth_pred

class DepthEnDecoder(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(DepthEnDecoder, self).__init__()
        # 7-layer encoder (Conv-GroupNorm-LeakyReLU, Conv-LeakyReLU, Conv-ReLU)
        encoder_network = [
            nn.Conv2d(cin, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14 -> 7x7
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=7, stride=1, padding=0, bias=False),  # 7x7 -> 1x1
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=1, stride=1, padding=0, bias=False),  # 1x1 -> 1x1
            nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*encoder_network)
        
        # decoder -> same as unsup3d
        decoder_network = [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            decoder_network += [activation()]
        
        self.decoder = nn.Sequential(*decoder_network)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x