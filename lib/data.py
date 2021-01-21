import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import cv2

import pandas as pd
import numpy as np

import os


class CatsDataset(Dataset):
    def __init__(self, index_file, img_dir, transform=None):
        self.names = pd.read_csv(index_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.names.iloc[idx, 0])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


DATA_DIR = "/home/v-eliseev/Datasets/cats/"
def makeCatsDataset(batch=16, isize=64, path=DATA_DIR):
    cats_dataset = CatsDataset(
        index_file = path+"faces_index.txt",
        img_dir = path+"faces/",
        transform = transforms.Compose([
            transforms.Pad(30, padding_mode='edge'),
            transforms.RandomAffine(
                degrees = 35
            ),
            transforms.Resize((isize,isize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    dataloader = DataLoader(cats_dataset, batch_size=batch, shuffle=True, num_workers=8)
    return dataloader