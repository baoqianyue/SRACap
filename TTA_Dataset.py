import os
import sys
import json
import pickle
import torch
from PIL import Image
from typing import Tuple


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class COCOCapDatasetForEmbedding(torch.utils.data.Dataset):
    def __init__(self, annotations: str, images_root=None, transform=None, split='train'):
        super().__init__()
        with open(annotations, 'r') as f:
            data = json.load(f)

        data = dict_slice(data, 0, 10)
        self.data = data
        self.filenames = list(self.data.keys())
        self.images_root = images_root
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        image_id = self.filenames[item]
        captions = self.data[image_id]

        filename = os.path.join(self.images_root, image_id)
        image = Image.open(filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id, captions, filename

def collate(batch):
    batch_size = len(batch)
    image, image_id, captions, filename = zip(*batch)

    image = torch.stack(image)  
   
    return image, image_id, captions, filename
