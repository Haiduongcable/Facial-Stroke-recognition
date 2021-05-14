import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torchvision.models as models
import numpy as np
from efficientnet_pytorch import EfficientNet
import torchvision
from utils import set_parameter_required_grad

class EfficientNet_Facial(nn.Module):
    def __init__(self, num_class, use_pretrained = True):
        super(EfficientNet_Facial, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_class = num_class
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 2).to(self.device)
        self.softmax = nn.Softmax(dim = 1).to(self.device)
#         self.dense = nn.Sequential(nn.Linear(in_features = 1000, out_features = self.num_class), nn.Softmax(dim = 1))
    def forward(self, image):
        output = self.backbone(image)
#         output = self.dense(output)
        output = self.softmax(output)
        return output

    def freezebackbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone._fc.bias.requires_grad = True
        self.backbone._fc.weight.requires_grad = True
    
    def finetune_alllayer(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = np.zeros((10,3, 224,224), dtype = np.float32)
    print(type(image))
    image = torch.from_numpy(image).to(device)
    model = EfficientNet_Facial(2)
    print(model)
    output = model(image)
    print(np.shape(output))
    print(output)
#     for name, param in model.named_parameters():
#         print(name, param.size())
#     model.backbone._fc.bias.requires_grad = False
#     model.backbone._fc.weight.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False
    model.backbone._fc.bias.requires_grad = True
    model.backbone._fc.weight.requires_grad = True
#     count  = 0
#     for child in model.children():
#         for layer in child:
#             print(layer)
#     for param in model.parameters():
# #         param.requires_grad = False
#         print(param)
    