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

def plot_sample(all_samples, item_in_batch, side_size=32):
    
    frames_count = all_samples['images'].size()[0]
    
    fig = plt.figure(figsize=(side_size, side_size))
    columns = frames_count
    rows = 1
    
    print(all_samples['description'][item_in_batch])
    
    # The batch index is second, so I need a little more complicated way to get the 
    # samle description from the batch (the same for image frames)
    encoded_description = []
    enc_desc_length = all_samples['desc_encoded'].size(0)
    for word_idx in range(enc_desc_length):
        encoded_description.append(all_samples['desc_encoded'][word_idx][item_in_batch].item())
        
    print(encoded_description)
    
    for frame_idx in range(frames_count):
        plot_at(all_samples['images'][frame_idx][item_in_batch], frame_idx + 1, fig, rows, columns)
    
    plt.show()

def plot_at(sample_to_show, subplot_index, fig, rows, columns):
    x = sample_to_show
    x = x.permute(1, 2, 0)

    fig.add_subplot(rows, columns, subplot_index)
    plt.imshow(x.cpu())
    
def plot_samples(samples, demos_to_show, side_size=32):
    for i in range(demos_to_show):
        print(i)
        plot_sample(samples, i)
        
def plot_interpolations(first_frame, interpolation_real, last_frame, interpolation, description, demos_to_show):
    for i in range(demos_to_show):
        print(i)
        plot_interpolation(first_frame[i], interpolation_real[i], last_frame[i], interpolation[i], description[i])
        
def plot_interpolation(first_frame, interpolation_real, last_frame, interpolation, description, side_size=32):
    print(description)
    
    fig = plt.figure(figsize=(side_size, side_size))
    rows = 1
    columns = 4
    
    plot_at(first_frame, 1, fig, rows, columns)
    plot_at(interpolation_real, 2, fig, rows, columns)
    plot_at(last_frame, 3, fig, rows, columns)
    plot_at(interpolation, 4, fig, rows, columns)
    
    plt.show()
    
def plot_edge_frames(reference, start_frame, end_frame, start_frame_fake, end_frame_fake, description, demos_to_show):
    for i in range(demos_to_show):
        print(i)
        plot_edge_frame(reference[i], start_frame[i], end_frame[i], start_frame_fake[i], end_frame_fake[i], description[i])
    
def plot_edge_frame(reference, start_frame, end_frame, start_frame_fake, end_frame_fake, description, side_size=32):
    print(description)
    
    fig = plt.figure(figsize=(side_size, side_size))
    rows = 1
    columns = 5
    
    plot_at(reference, 1, fig, rows, columns)
    plot_at(start_frame, 2, fig, rows, columns)
    plot_at(end_frame, 3, fig, rows, columns)
    plot_at(start_frame_fake, 4, fig, rows, columns)
    plot_at(end_frame_fake, 5, fig, rows, columns)
    
    plt.show()