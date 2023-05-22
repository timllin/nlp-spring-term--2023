import gdown
import os
import shutil
from PIL import Image
import torch
from torch.utils.data import Dataset

def download(url):
    gdown.download_folder(url, quiet=True, use_cookies=False, remaining_ok=True)


def moving_data(path):
    new_dir = os.path.join(path+"images")
    os.mkdir(new_dir)
    for subdir, dirs, files in os.walk(path):
        for file in files:
            shutil.copy(os.path.join(subdir, file), os.path.join(new_dir, file))

class DiffusiontDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        return image
