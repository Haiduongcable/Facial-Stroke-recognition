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
from dataset_landmark import Face_LandMarkDataset
from transform_pairimage import ToTensor, Pair_Resize
from tqdm import tqdm


class Dataloader():
    def __init__(self, batch_size = 64, num_workers = 8):
        self.train_dir_facial = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Data/Data/train"
        self.val_dir_facial = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Data/Data/val"


        self.train_dir_landmark = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Data/Data/train_landmark_image"
        self.val_dir_landmark = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Data/Data/val_landmark_image"

        self.img_size_org = 256
        self.img_size = 224
        self.device = 'cuda'
        self.use_cuda = torch.cuda.is_available()
        self.batch_size = batch_size


    def create_transform(self):
        scale = Pair_Resize((224,224))
        train_transform = transforms.Compose([Pair_Resize((224,224)), ToTensor()])
        test_transforms = transforms.Compose([Pair_Resize((224,224)), ToTensor()])
        return train_transform, test_transforms

    def get_loader(self):
        kwargs = {'num_workers': 8, 'pin_memory': True} if self.use_cuda else {}
        train_transforms, val_transforms = self.create_transform()
        #Initial dataset
        train_dataset = Face_LandMarkDataset(dir_facial_image= self.train_dir_facial,
                                           dir_landmark_image= self.train_dir_landmark,
                                           transform= train_transforms)
        val_dataset = Face_LandMarkDataset(dir_facial_image= self.val_dir_facial,
                                           dir_landmark_image= self.val_dir_landmark,
                                           transform= val_transforms)
        
        #Initial data loader 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        print('Len Dataset:  ', len(train_dataset), len(val_dataset))
        return train_loader, val_loader, train_dataset, val_dataset

if __name__ == '__main__':
    loader = Dataloader()
    train_loader, val_loader, train_dataset, val_dataset = loader.get_loader()
    for i, sample_batch in enumerate(tqdm(val_loader)):
        facial_data, landmark_data, label = sample_batch["facial_image"],\
                                             sample_batch["landmark_image"], sample_batch["label"]
        # print(facial_data)
    