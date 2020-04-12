import torch
from torch.utils.data import Dataset
from PIL import Image
from src.augmentation import transform
from src.data.voc_utils import *

class VOCxx(Dataset):
    """
        PASCAL VOC data loader based on PyTorch dataset class for batch processing
    """
    def __init__(self, split, dataroot, datayears, datanames, discard_difficult=True, use_augment=True):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.use_augment = use_augment
        self.discard_difficult = discard_difficult

        # Get datas
        self.dataroot, self.datayears, self.datanames = dataroot, datayears, datanames        
        self.image_lists, self.target_lists = parse_data(self.dataroot, self.datanames)
        assert len(self.image_lists) == len(self.target_lists)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.image_lists[i], mode='r')
        image = image.convert('RGB')

        # Read targes
        targets = parse_annotation(self.target_lists[i]) 

        # Discard difficult objects, if desired
        if self.discard_difficult:
            keep_idx = targets[:, -1] == 0
            targets = targets[keep_idx, :]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, targets[:, :4], targets[:, 4], targets[:, 5], self.split, self.use_augment)
        targets = torch.cat((boxes, labels.unsqueeze(1).float(), difficulties.unsqueeze(1).float()), dim=1)
        return image, targets

    def __len__(self):
        return len(self.image_lists)

    def collate_fn(self, batch):
        images, boxes = list(), list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
        images = torch.stack(images, dim=0)
        return images, boxes
