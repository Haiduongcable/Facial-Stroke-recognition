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



def set_parameter_required_grad(model, required_grad):
    for param in model.parameters():
        param.required_grad = required_grad


def evaluate(output, label):
    c_True_Positive = torch.sum((output == 1) & ((label == 1)))
    c_False_Positive = torch.sum((output == 1) & ((label == 0)))
    c_True_Negative = torch.sum((output == 0) & ((label == 0)))
    c_False_Negative = torch.sum((output == 0) & ((label == 1)))
    c_Positive = torch.sum((label == 1))
    c_Negative = torch.sum((label == 0))
    return c_True_Positive, c_False_Positive, c_True_Negative, c_False_Negative, c_Positive, c_Negative

def eval_numpy(output, label):
    c_True_Positive = np.sum((output == 1) & ((label == 1)))
    c_False_Positive = np.sum((output == 1) & ((label == 0)))
    c_True_Negative = np.sum((output == 0) & ((label == 0)))
    c_False_Negative = np.sum((output == 0) & ((label == 1)))
    c_Positive = np.sum((label == 1))
    c_Negative = np.sum((label == 0))
    return c_True_Positive, c_False_Positive, c_True_Negative, c_False_Negative, c_Positive, c_Negative

def get_class_idx():
    class_to_idx= {'negative': 0, 'positive': 1}
    print(class_to_idx)
    idx_to_class={idx:cls for cls,idx in class_to_idx.items()}
    print(idx_to_class)
    return class_to_idx, idx_to_class

    # F1 = (True_Positive * False_Positive)
    # Recall = 
    # Precision = 
    # acc = (output.argmax(dim=1) == label).float().mean()
# def create_criterion(class):
