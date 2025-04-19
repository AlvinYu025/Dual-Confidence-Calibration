import torch
import numpy as np


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

        try:
            self.targets = np.array(dataset.label_list)[self.indices]
        except:
            pass

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)

        try:
            return im, targets
        except:
            return im

    def __len__(self):
        return len(self.indices)


class SingleClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_class):
        self.dataset = dataset
        self.indices = np.where(np.array(dataset.label_list) == target_class)[0]
        self.targets = np.array(dataset.label_list)[self.indices]
        self.target_class = target_class

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)


class ClassSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_classes):
        self.dataset = dataset
        self.indices = np.where(
            np.isin(np.array(dataset.label_list), np.array(target_classes)))[0]
        self.targets = np.array(dataset.label_list)[self.indices]
        self.target_classes = target_classes

    def __getitem__(self, idx):
        im, targets = self.dataset[self.indices[idx]]
        return im, targets

    def __len__(self):
        return len(self.indices)
