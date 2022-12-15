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

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_word, hidden_size, rnn_layers_count):
        super(TextEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn_layers_count = rnn_layers_count
        
        self.lookup = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_word)
        self.rnn = nn.RNN(embedding_dim, hidden_size, rnn_layers_count)
        
    def forward(self, input_descriptions):
        
        batch_size = input_descriptions.size()[1]
        h0 = torch.zeros(self.rnn_layers_count, batch_size, self.hidden_size).to(self.device())
                
        embedded_sequence = self.lookup(input_descriptions)
        
        _, h = self.rnn(embedded_sequence, h0)
        
        return h

    def device(self):
        return next(self.parameters()).device

class UnetEncoder(nn.Module):
    def __init__(self, image_channels, hidden_channels, rnn_hidden_layer_size):
        super(UnetEncoder, self).__init__()
        self.upfeature = FeatureMapBlock(image_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, rnn_hidden_layer_size) # 16 channels, contract * 4
        #self.fc = nn.Linear(4096, rnn_hidden_layer_size) # 32 channels, contract*4
        #self.relu = nn.LeakyReLU()

    def forward(self, x):
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
        
        out = self.flatten(x3)
        out = self.fc(out)
        #out = self.relu(out)
        
        return out
    
class ImagesEncoder(nn.Module):
    def __init__(self, image_channels, hidden_channels, rnn_hidden_layer_size):
        super(ImagesEncoder, self).__init__()
        
        self.rnn_cell = ImagesRNNCell(image_channels, hidden_channels, rnn_hidden_layer_size)
        
    def forward(self, images, h0):
        
        h = h0
        outputs = []
        
        for _, frame in enumerate(images):
            _, h = self.rnn_cell(frame, h)
            outputs.append(h)
        
        out = torch.stack(outputs, dim=0)           
        
        return out, h
