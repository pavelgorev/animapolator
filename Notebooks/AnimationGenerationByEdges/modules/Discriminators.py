import torch
from torch import nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import numpy as np

import os
from PIL import Image

from modules.Encoders import TextEncoder, UnetEncoder, ImagesEncoder
from modules.ImageNetworkBlocks import ContractingBlock, ExpandingBlock, FeatureMapBlock, FeatureExchangeBlock

class UnifiedDiscriminator(nn.Module):
    def __init__(self, count_of_words, embedding_dim, padding_word, h_size, image_channels, hidden_channels):
        super(UnifiedDiscriminator, self).__init__()
        
        self.h_size = h_size
            
        self.text_encoder = TextEncoder(
            count_of_words, 
            embedding_dim, 
            padding_word, 
            h_size, 
            1)
        
        self.frames_encoder = ImagesEncoder(image_channels, hidden_channels, h_size)
        self.reference_encoder = UnetEncoder(image_channels, hidden_channels, h_size)
        
        self.fc = nn.Linear(h_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, description, reference, frames):
                
        text_h = self.text_encoder(description)
        # Size of h: [1, batches_size, layer_size]. The first dimension is layer index, there is only one layer now.
        text_h = text_h[0]
        
        reference_h = self.reference_encoder(reference)
        
        batch_size = reference.size()[0]
        h0 = torch.zeros(batch_size, self.h_size).to(self.device())
        
        _, frames_h = self.frames_encoder(frames, h0)
        
        concatenated_features = torch.cat([text_h, reference_h, frames_h], dim=1)
        out = self.fc(concatenated_features)
        
        return out

    def device(self):
        return next(self.parameters()).device
    
class AveragedSequenceDiscriminator(nn.Module):
    def __init__(self, count_of_words, embedding_dim, padding_word, h_size, image_channels, hidden_channels):
        super(AveragedSequenceDiscriminator, self).__init__()
        
        self.h_size = h_size
            
        self.text_encoder = TextEncoder(
            count_of_words, 
            embedding_dim, 
            padding_word, 
            h_size, 
            1)
        
        self.frames_encoder = ImagesEncoder(image_channels, hidden_channels, h_size)
        self.reference_encoder = UnetEncoder(image_channels, hidden_channels, h_size)
        
        self.fc = nn.Linear(h_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, description, reference, frames):
                
        text_h = self.text_encoder(description)
        # Size of h: [1, batches_size, layer_size]. The first dimension is layer index, there is only one layer now.
        text_h = text_h[0]
        
        reference_h = self.reference_encoder(reference)
        
        batch_size = reference.size()[0]
        h0 = torch.zeros(batch_size, self.h_size).to(self.device())
        
        frames_estimations, _ = self.frames_encoder(frames, h0)
        
        output = []
        for frame_estimation in frames_estimations:
            concatenated_features = torch.cat([text_h, reference_h, frame_estimation], dim=1)
            output.append(self.fc(concatenated_features))
        
        return torch.stack(output, dim=0)

    def device(self):
        return next(self.parameters()).device
    
class SequenceDiscriminator(nn.Module):
    def __init__(self):
        super(SequenceDiscriminator, self).__init__()
        
        return
        
    def forward(self, description, reference, frames):
        
        return 0

class ImageDiscriminator(nn.Module):
    def __init__(self, h_size, image_channels, hidden_channels):
        super(ImageDiscriminator, self).__init__()
        
        self.image_encoder = UnetEncoder(image_channels, hidden_channels, h_size)
        
        return
        
    def forward(self, frame):
        
        estimate = self.image_encoder(frame)
               
        return estimate
    
class ImageDiscriminatorUnflattened(nn.Module):
    def __init__(self, image_channels, hidden_channels):
        super(ImageDiscriminatorUnflattened, self).__init__()
        self.upfeature = FeatureMapBlock(image_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=False)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=False)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=False)
        self.downfeature = FeatureMapBlock(hidden_channels * 8, image_channels)
        
    def forward(self, x):
        
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        out = self.downfeature(x3)
        
        return out
    
class InterpolationDiscriminator(nn.Module):
    def __init__(self, count_of_words, embedding_dim, padding_word, rnn_hidden_layer_size, image_channels, hidden_channels):
        super(InterpolationDiscriminator, self).__init__()
            
        self.text_encoder = TextEncoder(
            count_of_words, 
            embedding_dim, 
            padding_word, 
            rnn_hidden_layer_size, 
            1)
        
        self.upfeature = FeatureMapBlock(3 * image_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        #self.contract3 = ContractingBlock(hidden_channels * 4)
        #self.contract4 = ContractingBlock(hidden_channels * 8)
        #self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
        self.final = nn.Conv2d(hidden_channels * 4, 1, kernel_size=1)
        
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU(0.2)
        
        fc_size = rnn_hidden_layer_size + 64
        
        self.fc = nn.Linear(rnn_hidden_layer_size + 64, fc_size)
        self.fc1 = nn.Linear(fc_size, fc_size)
        self.relu1 = nn.LeakyReLU(0.2)
        
    def forward(self, description, first_frame, last_frame, interpolation):
        
        text_h = self.text_encoder(description)
        # Size of h: [1, batches_size, layer_size]. The first dimension is layer index, there is only one layer now.
        text_h = text_h[0]
        
        x = torch.cat([first_frame, last_frame, interpolation], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        #x3 = self.contract3(x2)
        #x4 = self.contract4(x3)
        xn = self.final(x2)
        xfl = self.flatten(xn)
        #x_resized = self.images_fc(xfl)
        #x_out = self.relu(x_resized)
        
        combined = torch.cat([text_h, xfl], axis=1)
        out = self.fc(combined)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        
        return out

    def device(self):
        return next(self.parameters()).device

class ThreeImagesAndDescriptionDiscriminator(nn.Module):
    def __init__(self, count_of_words, embedding_dim, padding_word, rnn_hidden_layer_size, image_channels, hidden_channels):
        super(ThreeImagesAndDescriptionDiscriminator, self).__init__()
            
        self.interpolation_discriminator = InterpolationDiscriminator( 
            count_of_words, 
            embedding_dim, 
            padding_word, 
            rnn_hidden_layer_size, 
            image_channels,
            hidden_channels)
        
    def forward(self, description, first_frame, last_frame, interpolation):
        
        interpolation_estimate = self.interpolation_discriminator(description, first_frame, last_frame, interpolation)
        
        return interpolation_estimate

    def device(self):
        return next(self.parameters()).device 
    