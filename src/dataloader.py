import os 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import cv2
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm


class DataLoader():
    def __init__(self):
        self.train_dir = "../Data/train"
        self.test_dir = "../Data/val"
        self.img_size_org = 256
        self.img_size = 224
        self.device = 'cuda'
        self.use_cuda = torch.cuda.is_available()
        self.batch_size = 32
        

    def create_transform(self):
        train_transforms = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            # transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        test_transforms = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            #transforms.RandomHorizontalFlip(),for i_batch, sample_batched in enumerate(test_dataset):
    #     print(np.shape(sample_batched))
    # # loader.get_weight_class(train_dataset)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return train_transforms, test_transforms
    def get_loader(self):
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.use_cuda else {}
        train_transforms, test_transforms = self.create_transform()
        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        test_dataset = datasets.ImageFolder(root=self.test_dir, transform=test_transforms)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, **kwargs) 
        print(len(train_dataset), len(test_dataset))
        return train_loader, test_loader, train_dataset, test_dataset
    def get_weight_class(self, train_dataset):
        (unique, counts) = np.unique(train_dataset.targets, return_counts=True)
        cw=1/counts
        cw/=cw.min()
        class_weights = {i:cwi for i,cwi in zip(unique,cw)}
        print(counts, class_weights.values())
        return class_weights
    
if __name__ == '__main__':
    loader = DataLoader()
    train_loader, test_loader, train_dataset, test_dataset = loader.get_loader()
    for data, label in test_loader:
        print(label)
        print(np.shape(data))

        break
        
    # for i_batch, sample_batched in enumerate(test_dataset):
    #     print(np.shape(sample_batched))
    # # loader.get_weight_class(train_dataset)
    # num_classes=len(train_dataset.classes)
    # print(num_classes)
        

