from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from transform_pairimage import Pair_Resize, ToTensor
import cv2


class Face_KeypointDataset(Dataset):
    def __init__(self, dir_facial_image, dir_keypoint, transform = None):
        """
        Args:
            dir_facial_image: path to folder all facial image
            dir_keypoint: path to folder all keypoint txt file 
            transform(callable, optional): Optional transform to be applied on a sample
        """

        self.dir_facial_image = dir_facial_image
        self.dir_keypoint = dir_keypoint
        self.transform = transform
        self.list_annotation = ["positive", "negative"]
        # for folder in os.listdir(self.dir_landmark_image):
        #     self.list_annotation.append(folder)
        
        self.list_path_facial_image = []
        self.list_label = []
        self.list_path_keypoint = []
        for folder in os.listdir((self.dir_keypoint)):
            path_folder = self.dir_facial_image + "/" + folder
            path_folder_keypoint = self.dir_keypoint + "/" + folder
            for file in os.listdir(path_folder_keypoint):
                name_image = file[:-3] + "jpg"
                path_facialimage = path_folder + "/" + name_image 
                path_keypoint = path_folder_keypoint + "/" + file
                label = folder 
                self.list_path_facial_image.append(path_facialimage)
                self.list_path_keypoint.append(path_keypoint)
                self.list_label.append(label)

    
    def __len__(self):
        len_dataset = len(self.list_path_keypoint)
        return len_dataset
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        labels_to_index = {}
        for i, label in enumerate(self.list_annotation):
            labels_to_index[label] = i

        path_facial = self.list_path_facial_image[index]
        path_keypoint = self.list_path_keypoint[index]
        label = self.list_label[index]
        facial_image = cv2.imread(path_facial)
        keypoint = 
        label = labels_to_index[label]
        sample = {'facial_image': facial_image, 'landmark_image': landmark_image,'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample