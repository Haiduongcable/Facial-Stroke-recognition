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
from utils import set_parameter_required_grad


class Efficient_FacialNet(nn.Module):
    def __init__(self, num_class, use_pretrained = True):
        super(Efficient_FacialNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.num_class = num_class
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
        self.model.classifier = torch.nn.Identity()
        # path_pretrained = "../Save_checkpoint/enet_b0_8_best_afew.pt"
        if use_pretrained:
            path_pretrained = "../models/pretrained_faces/state_vggface2_enet0_new.pt"

            if not torch.cuda.is_available():
                self.model.load_state_dict(torch.load(path_pretrained,map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(path_pretrained))
        

        self.model.classifier = nn.Sequential(nn.Linear(in_features = 1280, out_features = self.num_class), nn.Softmax(dim = 1))
        self.model = self.model.to(self.device)
        # self.softmax = torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
    
    def forward(self, x):
        output = self.model(x)
        # output = nn.Softmax(output)
        return output
    
    def freezebackbone(self):
        set_parameter_required_grad(self.model, required_grad= False)
        set_parameter_required_grad(self.model.classifier, required_grad = True)
    def finetune_alllayer(self):
        set_parameter_required_grad(self.model, required_grad= True)

    
        
if __name__ == '__main__':
    x = np.zeros((1,224,224,3))
    x = torch.from_numpy(x).to("cpu")
    model = Efficient_FacialNet(10)
    print(model)
    output = model(x)
    print(np.shape(output))
    
