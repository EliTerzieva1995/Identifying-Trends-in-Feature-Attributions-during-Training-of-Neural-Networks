from PIL import Image
from typing import Tuple, Any
import torch
from torch.utils.data import DataLoader  # type: ignore
from torchvision.datasets import VisionDataset  # type: ignore
import numpy as np


class CustomDataset(VisionDataset):
    def __init__(self, images, transform):
        super(CustomDataset, self).__init__('./')
        self.data = torch.stack(list(images['images'])).numpy()
        self.transform = transform
        self.targets = torch.tensor(images['label_int'])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.array(img).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        """ Get length of dataset """
        return len(self.data)