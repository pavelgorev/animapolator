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

def get_description_from(folder_name):
    first_space_index = folder_name.find(' ')
    return folder_name[first_space_index + 1:]

def get_text_parameters(tokenizer, all_descriptions):
    def yield_tokens(desc_list):
        for desc in desc_list:
            yield tokenizer(desc)

    vocab = build_vocab_from_iterator(yield_tokens(all_descriptions))
    padding_word = len(vocab)

    encode = lambda x: vocab(tokenizer(x))

    count_of_words=len(vocab) + 1 # including a padding word - that's why there is "+ 1"
    
    return padding_word, encode, count_of_words
