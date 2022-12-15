from modules.ImageNetworkBlocks import ContractingBlock, ExpandingBlock, FeatureMapBlock, FeatureExchangeBlock

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

class ImagesRNNCell(nn.Module):
    def __init__(self, image_channels, hidden_channels, rnn_hidden_layer_size):
        super(ImagesRNNCell, self).__init__()
        self.upfeature = FeatureMapBlock(image_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        
        self.feature_exchange = FeatureExchangeBlock(2048, (128, 4, 4), rnn_hidden_layer_size)   # 16 channels, contract * 4
        #self.feature_exchange = FeatureExchangeBlock(4096, (256, 4, 4), rnn_hidden_layer_size)  # 32 channels, contract * 4

        self.expand3 = ExpandingBlock(hidden_channels * 8)
        self.expand4 = ExpandingBlock(hidden_channels * 4)
        self.expand5 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, image_channels)
        self.limiter = torch.nn.Tanh()

    def forward(self, x, h):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
                
        # flatten -> concat with h -> feature exchange -> rnn -> (extract h) + (unflatten)
        x_exchanged, h_new = self.feature_exchange(x3, h)
        
        x10 = self.expand3(x_exchanged, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.limiter(xn), h_new