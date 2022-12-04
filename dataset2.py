import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from pathlib import Path
import pandas as pd
from PIL import Image


class ReIDDataset(Dataset):

    def __init__(self, path, transform=None):
        self.transform = transform
        self.df = pd.read_pickle(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx]['im_path'])

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'id': int(self.df.iloc[idx]['label']), 'im_path': self.df.iloc[idx]['im_path'], 'role': self.df.iloc[idx]['role']}

        return sample

