import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils

from PIL import Image
import cv2

import pandas as pd
import numpy as np

import os

DATA_DIR = "/home/v-eliseev/Datasets/cats/"
def makeCatsDataset(batch=16, isize=64, path=DATA_DIR):
    cats_dataset = ImageFolder(
        root = path,
        transform = transforms.Compose([
            transforms.Pad(isize//4, padding_mode='reflect'),
            transforms.RandomAffine(
                degrees = 10
            ),
            transforms.Resize((isize,isize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    dataloader = DataLoader(cats_dataset, batch_size=batch, shuffle=True, num_workers=8)
    return dataloader