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

class Face_LandMarkDataset(Dataset):
    def __init__(self, dir_facial_image, dir_landmark_image, transform = None):
        """
        Args:
            dir_facial_image: path to folder all facial image
            dir_landmark_image: path to folder all landmark mask facial image 
            transform(callable, optional): Optional transform to be applied on a sample
        """

        self.dir_facial_image = dir_facial_image
        self.dir_landmark_image = dir_landmark_image
        self.transform = transform
        self.list_annotation = []
        for folder in os.listdir(self.dir_landmark_image):
            self.list_annotation.append(folder)
        
        self.list_path_facial_image = []
        self.list_label = []
        self.list_path_landmark_image = []
        for folder in os.listdir((self.dir_facial_image)):
            path_folder = self.dir_facial_image + "/" + folder
            path_folder_landmark = self.dir_landmark_image + "/" + folder
            for file in os.listdir(path_folder_landmark):
                path_facialimage = path_folder + "/" + file 
                path_landmarkimage = path_folder_landmark + "/" + file
                label = folder 
                self.list_path_facial_image.append(path_facialimage)
                self.list_path_landmark_image.append(path_landmarkimage)
                self.list_label.append(label)

    
    def __len__(self):
        len_dataset = len(os.listdir(self.dir_facial_image))
        return len_dataset
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        labels_to_index = {}
        for i, label in enumerate(self.list_annotation):
            labels_to_index[label] = i

        path_facial = self.list_path_facial_image[index]
        path_landmark = self.list_path_landmark_image[index]
        label = self.list_label[index]
        facial_image = io.imread(path_facial)
        landmark_image = io.imread(path_landmark)
        label = labels_to_index[label]
        sample = {'facial_image': facial_image, 'landmark_image': landmark_image,'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    dir_facial_image = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Data/val"
    dir_landmark_image = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Data/val_landmark_image"
    
    scale = Pair_Resize((224,224))
    composed = transforms.Compose([scale])

    transformed_dataset = Face_LandMarkDataset(dir_facial_image=dir_facial_image,
                                           dir_landmark_image=dir_landmark_image,
                                           transform= composed)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, np.shape(sample['facial_image']), np.shape(sample['landmark_image']), sample['label'])

        

    # fig = plt.figure()

    # sample = face_dataset[65]
    # for i, tsfrm in enumerate([scale, crop, composed]):
    #     transformed_sample = tsfrm(sample)

    #     ax = plt.subplot(1, 3, i + 1)
    #     plt.tight_layout()
    #     ax.set_title(type(tsfrm).__name__)
    #     show_landmarks(**transformed_sample)

    # plt.show()
