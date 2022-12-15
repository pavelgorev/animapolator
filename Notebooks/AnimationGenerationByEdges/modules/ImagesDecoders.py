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

from modules.ImagesRNNCell import ImagesRNNCell
from modules.ImageNetworkBlocks import ContractingBlock, ExpandingBlock, FeatureMapBlock, FeatureExchangeBlock

class ImagesDecoderFullPath(nn.Module):
    def __init__(self, image_channels, hidden_channels, rnn_hidden_layer_size, animation_limit):
        super(ImagesDecoderFullPath, self).__init__()
        
        self.animation_limit = animation_limit
        
        self.rnn_cell = ImagesRNNCell(image_channels, hidden_channels, rnn_hidden_layer_size)
        
    def forward(self, reference_image, h):
        
        should_stop = False
        output = []
        frames_count = 0
        
        while not should_stop:
            new_frame, h = self.rnn_cell(reference_image, h)
            output.append(new_frame)
            
            frames_count = frames_count + 1
            
            new_frame_is_all_zeros = torch.count_nonzero(new_frame).item() == 0
            should_stop = new_frame_is_all_zeros or (frames_count >= self.animation_limit)
        
        return torch.stack(output)

class ImagesDecoderDirectForward(nn.Module):
    def __init__(self, image_channels, hidden_channels, rnn_hidden_layer_size, animation_limit):
        super(ImagesDecoderDirectForward, self).__init__()
        
        self.animation_limit = animation_limit
        
        #self.rnn_cell = ImagesRNNCell(image_channels, hidden_channels, rnn_hidden_layer_size)
        self.rnn_cell = nn.RNNCell(rnn_hidden_layer_size, rnn_hidden_layer_size)

        self.upfeature = FeatureMapBlock(image_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        #self.contract4 = ContractingBlock(hidden_channels * 8, use_dropout=True)

        self.feature_exchange = FeatureExchangeBlock(2048, (128, 4, 4), rnn_hidden_layer_size)  #16 channels, contract*4
        #self.feature_exchange = FeatureExchangeBlock(1024, (256, 2, 2), rnn_hidden_layer_size)  #16 channels, contract*8
        #self.feature_exchange = FeatureExchangeBlock(2048, (512, 2, 2), rnn_hidden_layer_size) #32 channels; contract*8

        #self.expand2 = ExpandingBlock(hidden_channels * 16)
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, image_channels)
        self.limiter = torch.nn.Tanh()
        
    def forward(self, reference_image, h):
        
        should_stop = False
        output = []
        frames_count = 0
        
        while not should_stop:
            h = self.rnn_cell(torch.zeros_like(h), h)

            new_frame = self.decode_image(reference_image, h)
            output.append(new_frame)
            
            frames_count = frames_count + 1
            
            new_frame_is_all_zeros = torch.count_nonzero(new_frame).item() == 0
            should_stop = new_frame_is_all_zeros or (frames_count >= self.animation_limit)
        
        return torch.stack(output)

    def decode_image(self, reference_image, h):
        x0 = self.upfeature(reference_image)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        #x4 = self.contract4(x3)
                
        # flatten -> concat with h -> feature exchange -> rnn -> (extract h) + (unflatten)
        x_exchanged, h_new = self.feature_exchange(x3, h)
        
        #x9 = self.expand2(x_exchanged, x3)
        x10 = self.expand3(x_exchanged, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)

        return self.limiter(xn)
    
class InterpolationDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, rnn_hidden_layer_size):
        super(InterpolationDecoder, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)

        self.feature_exchange = FeatureExchangeBlock(2048, (128, 4, 4), rnn_hidden_layer_size)
    
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.limiter = torch.nn.Sigmoid()

    def forward(self, first_frame, last_frame, h):

        batch_size = first_frame.size(0)
        
        x = torch.cat([first_frame, last_frame], axis=1)
        
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        #x4 = self.contract4(x3)
        #x5 = self.contract5(x4)
        #x6 = self.contract6(x5)
        
        xfc, h_new = self.feature_exchange(x3, h)     
        
        #x7 = self.expand0(x6, x5)
        #x8 = self.expand1(x5, x4)
        #x9 = self.expand2(x4, x3)
        x10 = self.expand3(xfc, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.limiter(xn)
    
class ContractingPath(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, rnn_hidden_layer_size):
        super(ContractingPath, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)

    def forward(self, reference, h):
        
        x0 = self.upfeature(reference)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)

        return x0, x1, x2, x3
    
class ExpandingPath(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, rnn_hidden_layer_size):
        super(ExpandingPath, self).__init__()
        
        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.limiter = torch.nn.Sigmoid()
        
    def forward(self, x0, x1, x2, xfc):
        x10 = self.expand3(xfc, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        
        return self.limiter(xn)
        