from torch.utils.data import Dataset
from processing import get_label
from tqdm import tqdm
from torchtext import data
from processing import process_img
import numpy as np
import torch

class DatasetWithImg(Dataset):
    def __init__(self, df, crop_size, mode, verbose=False):
        self.mode = mode
        self.text = df['text']
        self.images = df['image']
        self.event_label = df['event']
        if self.mode == 'train_pictures':
            self.label = df['label']
        self.crop_size = crop_size
        self.verbose = verbose
        self.mask = df['mask']

    def __len__(self):
        if self.mode == 'train_pictures':
            return len(self.label)
        else:
            return len(self.text)

    def __getitem__(self, id):
        img = process_img(self.mode, self.images[id], self.crop_size)
        if self.mode == 'train_pictures':
            return (np.array(self.text[id]), img, self.mask[id]), self.label[id], self.event_label[id]
        else:
            return (np.array(self.text[id]), img, self.mask[id]), self.event_label[id]

class DatasetWithoutImg(data.Dataset):
    def __init__(self, df, mode, verbose=False):
        self.mode = mode
        self.text = df['text']
        self.event_label = df['event']
        if self.mode == 'train_pictures':
            self.label = df['label']
        self.verbose = verbose
        self.mask = df['mask']

    def __len__(self):
        if self.mode == 'train_pictures':
            return len(self.label)
        else:
            return len(self.text)

    def __getitem__(self, id):
        if self.mode == 'train_pictures':
            return (np.array(self.text[id]), self.mask[id]), self.label[id], self.event_label[id]
        else:
            return (np.array(self.text[id]), self.mask[id]), self.event_label[id]
