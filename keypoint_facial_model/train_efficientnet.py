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

def train(model,n_epochs, learningrate, train_loader, test_loader, use_sam=False):
    # optimizer
    if use_sam:
        optimizer = SAM(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=learningrate)
    else:
        optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    best_acc=0
    best_model=None
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            if use_sam:
                #optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
  
                # second forward-backward pass
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

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
                epoch_val_accuracy += acc / len(test_loader)
                epoch_val_loss += val_loss / len(test_loader)
                c_True_Positive, c_False_Positive, c_True_Negative, c_False_Negative, c_Positive, c_Negative = evaluate(val_output, label)
                epoch_TP += c_True_Positive
                epoch_FP += c_False_Positive
                epoch_TN += c_True_Negative
                epoch_FN += c_False_Negative
                epoch_Positive += c_Positive
                epoch_Negative += c_Negative
            Recall = (epoch_TP)/(epoch_TP + epoch_FN)
            Precision = (epoch_TP)/(epoch_TP + epoch_FP)
            F1 = (2*(Recall * Precision))/(Recall + Precision)


        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        print("Recall: {Recall:.4f},  Precision: {Precision:.4f}, F1 Score: {F1:.4f}")
        if best_acc<epoch_val_accuracy:
            best_acc=epoch_val_accuracy
            best_model=copy.deepcopy(model.state_dict())
        #scheduler.step()
    
    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Best acc:{best_acc}")
        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(test_loader)
                epoch_val_loss += val_loss / len(test_loader)

        print(
            f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    else:
        print(f"No best model Best acc:{best_acc}")


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
    #weights = torch.FloatTensor(list(class_weights.values())).cuda()
    # weights = torch.FloatTensor(list(class_weights.values()))
    global criterion
    criterion = nn.CrossEntropyLoss()

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
    train(model ,3 ,0.001 ,train_loader ,test_loader ,use_sam=True)

    #Fine all layer
    model.finetune_alllayer()
    train(model ,3 ,0.001 ,train_loader ,test_loader ,use_sam=True)

    PATH= checkpoint_dir + '/enet_b0_8_best_afew.pt'
    model_name='enet0_8_pt'
    torch.save(model, PATH)


if __name__ == '__main__':
    main()

    

