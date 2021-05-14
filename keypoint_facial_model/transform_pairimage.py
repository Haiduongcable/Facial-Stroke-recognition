from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image 
import torch.nn.functional as F


class Pair_Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        facial_image,  landmark_image = sample['facial_image'], sample['landmark_image']
        label = sample['label']

        resize_f_img = transform.resize(facial_image, self.output_size)
        resize_l_img = transform.resize(landmark_image, self.output_size)

        # if (len(np.shape(resize_f_img)) < 3):
        #     facial_np = resize_f_img
        #     mask = np.zeros((np.shape(facial_np)[0], np.shape(facial_np)[1], 3), dtype = np.uint8)
        #     mask[:,:,0], mask[:,:,1], mask[:,:,2] = facial_np, facial_np, facial_np
        #     resize_f_img = mask

        # elif (np.shape(resize_f_img)[2] > 3):
        #     facial_np = resize_f_img
        #     mask = np.zeros((np.shape(facial_np)[0], np.shape(facial_np)[1], 3), dtype = np.uint8)
        #     mask = facial_np[:,:,:3]
        #     resize_f_img = mask
        # print(np.shape(resize_f_img), np.shape(resize_l_img))


        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'facial_image': resize_f_img, 'landmark_image': resize_l_img,'label': label}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        facial_image,  landmark_image = sample['facial_image'], sample['landmark_image']
        label = sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # facial_image = F.to_tensor(facial_image)
        # landmark_image = F.to_tensor(landmark_image)
        # if facial_image
        # print(np.shape(facial_image))
        # print("Facial image:  ",np.shape(facial_image))
        facial_image = facial_image.transpose((2,0,1))

        landmark_image = landmark_image.transpose((2,0,1))
        return {'facial_image': torch.from_numpy(facial_image),
                'landmark_image': torch.from_numpy(landmark_image),
                'label': label}



# class Normalize(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __init__(self, mean, std, inplace = False):
#         self.mean = mean 
#         self.std = std
#         self.inplace = inplace

#     def __call__(self, sample):
#         facial_image,  landmark_image = sample['facial_image'], sample['landmark_image']
#         label = sample['label']
#         facial_image = F.normalize(facial_image, self.mean, self.std)
#         landmark_image = F.normalize(landmark_image, self.mean, self.std)
        
#         return {'facial_image': facial_image,
#                 'landmark_image': landmark_image,
#                 'label': label}