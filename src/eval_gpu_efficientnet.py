from sam import SAM 
import copy
import os 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import cv2
import os
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
from efficientnet import EfficientNet_Facial
from dataloader import DataLoader
from utils import set_parameter_required_grad, evaluate
from tqdm import tqdm

def train(model,n_epochs, learningrate, train_loader, test_loader, use_sam=False, train = False):
    
    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        epoch_Positive = 0
        epoch_Negative = 0
        epoch_TP = 0
        epoch_FP = 0
        epoch_TN = 0
        epoch_FN = 0
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)
            acc = (val_output.argmax(dim=1) == label).float().mean()
            #print("label:   ",   label)
            epoch_val_accuracy += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)
            #print(np.shape(val_output),np.shape(label))
            c_True_Positive, c_False_Positive, c_True_Negative, c_False_Negative, c_Positive, c_Negative = evaluate(val_output, label)
            epoch_TP += c_True_Positive
            epoch_FP += c_False_Positive
            epoch_TN += c_True_Negative
            epoch_FN += c_False_Negative
            epoch_Positive += c_Positive
            epoch_Negative += c_Negative
        print(f"Postive label:  {epoch_Positive}, Negative label: {epoch_Negative}")
        Recall = (epoch_TP)/(epoch_TP + epoch_FN)
        Precision = (epoch_TP)/(epoch_TP + epoch_FP)
        F1 = (2*(Recall * Precision))/(Recall + Precision)


    print(
        f"val_loss : {val_loss:.4f} - val_acc: {val_accuracy:.4f}\n"
    )
    print(f"Recall: {Recall:.4f},  Precision: {Precision:.4f}, F1 Score: {F1:.4f}")


def main():
    n_epochs = 40
    lr = 3e-5
    gamma = 0.7
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    #Dataloader
    loader = DataLoader()
    train_loader, test_loader, train_dataset, test_dataset = loader.get_loader()
    class_weights = loader.get_weight_class(train_dataset)
    
    #Initial loss function 
#     print("CLASS WEIGHT:  ",list(class_weights.values()))
    class_weight = [1.0,1.5]
    weights = torch.FloatTensor(class_weight).cuda()
#     weights = torch.FloatTensor(list(class_weights.values()))
    global criterion
    criterion = nn.CrossEntropyLoss(weights)

    #Initial model 
    num_class = len(train_dataset.classes)
    model = EfficientNet_Facial(num_class = num_class)
    print(model)
    #Checkpoint save
    checkpoint_dir = "../Save_checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    #Training
    #First Step Freeze Backbone, Finetune FC layer
    # set_parameter_required_grad(model, required_grad = False)
    # set_parameter_required_grad(model.classifier, required_grad = True)
    model.freezebackbone()
    train(model ,3 ,0.001 ,train_loader ,test_loader ,use_sam=False)

    #Fine all layer
    model.finetune_alllayer()
    train(model ,25 ,3e-5 ,train_loader ,test_loader ,use_sam=False)

    PATH= checkpoint_dir + '/Efficientnet_Facial.pt'
    model_name='Efficientnet_Facial'
    torch.save(model, PATH)


if __name__ == '__main__':
    main()

    

