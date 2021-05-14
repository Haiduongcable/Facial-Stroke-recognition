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
from utils import set_parameter_required_grad, get_class_idx, eval_numpy
from models import Efficient_FacialNet
from dataloader import DataLoader
from tqdm import tqdm
from PIL import Image
#Test dir 

def inference(model, test_dir, device, class_to_idx, idx_to_class,test_transforms):
    '''
    ::param: model
    ::param: image np.uint8 h * w * d
    ::param: conf np.float32
    ::param: class_list [Positive, Negative]
    ::param: device: cuda if torch.cuda.visible() else cpu
    '''
    threshold = 0.9

    y_val, y_pred = [],[]
    model.eval()
    for class_name in tqdm(os.listdir(test_dir)):
        print(class_name)
        if class_name in class_to_idx:
            class_dir = os.path.join(test_dir, class_name)
            y = class_to_idx[class_name]
            print(class_dir)
            for img_name in os.listdir(class_dir):
                # try:
                filepath = os.path.join(class_dir, img_name)
                img_cv2 = cv2.imread(filepath)
                print(filepath)
                print(np.shape(img_cv2))
                img = Image.fromarray(img_cv2)
                if (np.shape(img_cv2)[2] > 3):
                    img = np.array(img)
                    img = img[:,:,:3]
                    img = Image.fromarray(img)
                img_tensor = test_transforms(img)
                img_tensor.unsqueeze_(0)
                scores = model(img_tensor.to(device))
                # sm = nn.Softmax(dim = 1)
                # scores = sm(scores)
                print(scores)
                scores = scores[0].data.cpu().numpy()
                # print(scores)
                # y_score_val.append(scores)
                if scores[1] > threshold:
                    pred = 1
                else:
                    pred = 0
                y_pred.append(pred)
                y_val.append(y)

    
    y_pred = np.array(y_pred)
    y_val = np.array(y_val)
    print(np.shape(y_val), np.shape(y_pred))
    # evaluate(output, label)
    acc=100.0*(y_val==y_pred).sum()/len(y_val)
    print("Accuracy: ", acc)
    TP, FP, TN, FN, P, N = eval_numpy(y_pred, y_val)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2 * Recall * Precision / (Recall + Precision)
    print(f"Recall:  {Recall:.4f}, Precision:  {Precision:.4f}, F1:   {F1:.4f}")




    
    # print(acc)
    # y_train=np.array(train_dataset.targets)
    # for i in range(y_scores_val.shape[1]):
    #     _val_acc=(y_pred[y_val==i]==i).sum()/(y_val==i).sum()
    # print('%s %d/%d acc: %f' %(idx_to_class[i],(y_train==i).sum(),(y_val==i).sum(),100*_val_acc))

if __name__ == '__main__':
    test_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # num_class = 2
    # model = Efficient_FacialNet(num_class, use_pretrained= False)

    path_pretrained = "../Save_checkpoint/Efficientnet_Facial.pt"

    # #Load pretrained 
    # if torch.cuda.is_available():
    #     model.load_state_dict(torch.load(path_pretrained))
    # else:
    #     model.load_state_dict(torch.load(path_pretrained,map_location=torch.device('cpu')))
    model = torch.load(path_pretrained,map_location=torch.device('cpu'))
    model.eval()



    #test_dir = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Dataset/1_vs_all_Dataset/Data_04_05/Data/val"



    test_dir = "/home/haiduong/Documents/VIN_BRAIN/Stroke_Recognition/Test_Chuan/Google_image_test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_to_idx, idx_to_class = get_class_idx()
    inference(model, test_dir, device, class_to_idx, idx_to_class,test_transforms)
    
