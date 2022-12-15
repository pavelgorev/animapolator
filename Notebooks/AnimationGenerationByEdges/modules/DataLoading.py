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

from modules.TextProcessing import get_description_from

class LabeledAnimationLoader(torch.utils.data.Dataset):
    def __init__(self, root = None, transform = None, preprocessed_data_path = None, encode_fnc = None, animation_limit = None):
        
        if (not root == None) and (not transform == None) and (not encode_fnc == None):
            self.encode = encode_fnc
            self.animation_limit = animation_limit
            self.init_from_images(root, transform)
        else:
            self.init_from_data(preprocessed_data_path)
        
    def init_from_images(self, root, transform):
        self.root = root
        self.transform = transform
        
        self.folders = os.listdir(root)
        
        self.load_all_items()
        
    def init_from_data(self, preprocessed_data_path):
        self.items = torch.load(preprocessed_data_path)
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
    
    def load_all_items(self):
        length = len(self.folders)
        
        self.items = [None] * length
        
        for i in range(length):
            self.items[i] = self.load_item(i)
    
    def load_item(self, idx):
        
        # take folder by an index
        folder_name = self.folders[idx]
        
        description = get_description_from(folder_name)
        description_encoded = torch.tensor(self.encode(description), dtype=torch.int64)
        images = self.load_images_from(folder_name)
        
        return { 'description': description, 'desc_encoded': description_encoded, 'reference': images[0], 'images': images }
    
    def load_images_from(self, folder_name):
        
        folder_path = os.path.join(self.root, folder_name)
        files_in_folder = os.listdir(folder_path)
        files_in_folder_count = len(files_in_folder)
        
        # Limiting number of frames to avoid memory problems (for a while, on the current stage of development)
        files_in_folder_count_limited = min(files_in_folder_count, self.animation_limit)
        step = (files_in_folder_count - 1) // (files_in_folder_count_limited - 1)
        
        images = [None] * files_in_folder_count_limited
        
        for i in range(files_in_folder_count_limited):
            path_to_file = os.path.join(self.root, folder_name, files_in_folder[i * step])
            images[i] = self.load_image_from(path_to_file)
            
        return torch.tensor(np.stack(images))
            
    def load_image_from(self, image_file):
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = self.transform(image)
        
        return image
    
    def save(self, path):
        torch.save(self.items, path)
        
    def get_descriptions(self):
        return [item['description'] for item in self.items]
    
def create_collate_frames_fn(padding_word):
    
    def collate_frames(data):
        batch_descriptions = [x['description'] for x in data]

        batch_description_encoded = [x['desc_encoded'] for x in data]
        batch_description_encoded = nn.utils.rnn.pad_sequence(batch_description_encoded, padding_value=padding_word)

        batch_references = [x['reference'] for x in data]
        batch_references = torch.stack(batch_references)

        batch_images = [x['images'] for x in data]
        batch_images = nn.utils.rnn.pad_sequence(batch_images)

        return { 'description': batch_descriptions, 'desc_encoded': batch_description_encoded, 'reference': batch_references, 'images': batch_images }
    
    return collate_frames

def create_loaders(data_set, padding_word, batch_size, train_size, validation_size, ignore_size):
    train_set, val_set, ignore = torch.utils.data.random_split(data_set, [train_size, validation_size, ignore_size])

    collate_frames = create_collate_frames_fn(padding_word)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_frames)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=validation_size, shuffle=True, collate_fn=collate_frames)
    
    return train_loader, val_loader
    
