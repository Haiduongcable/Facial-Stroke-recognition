import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torchvision.models as models
import numpy as np
from efficientnet_pytorch import EfficientNet
import torchvision
from utils import set_parameter_required_grad

class Keypoint_Net(nn.Module):
    def __init__(self, num_class, use_pretrained = True):
        super(Keypoint_Net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_fc_1 =  2000
        self.feature_fc_2 = 2000
        self.feature_fc_3 = 1280

        self.rate_dropout = 0.25
        self.num_class = num_class
        self.backbone = EfficientNet.from_name('efficientnet-b0', num_classes=2, include_top = False, in_channels=3)

        self.dense_keypoint = nn.Sequential(nn.Linear(in_features = 468 * 2, out_features = self.feature_fc_1),\
                                            nn.Dropout(p=self.rate_dropout),\
                                            nn.Linear(in_features = self.feature_fc_1, out_features = self.feature_fc_2 ),\
                                            nn.Dropout(p = self.rate_dropout),\
                                            nn.Linear(in_features = self.feature_fc_2, out_features = self.feature_fc_3))

        self.dense_concat = nn.Sequential(nn.Linear(in_features = 1280 * 2, out_features = self.num_class), nn.Softmax(dim = 1))
        # self.dense_2 = nn.Sequential(nn.Linear(in_features = 1280, out_features = self.num_class), nn.Softmax(dim = 1))
#         self.dense = nn.Sequential(nn.Linear(in_features = 1000, out_features = self.num_class), nn.Softmax(dim = 1))
    def forward(self, facial_image, keypoint):
        feature_facial_image = self.backbone_1(facial_image)
        feature_landmark_image = self.dense_keypoint(keypoint)
        #Flatten  
        out_facial = feature_facial_image.view(-1, 1280)
        out_landmark = feature_landmark_image.view(-1, 1280)

        flatten_feature = torch.cat((out_facial, out_landmark), 1)
        output = self.dense(flatten_feature)
        return output

    def freezebackbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def finetune_alllayer(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     image = np.zeros((10,3, 224,224), dtype = np.float32)
#     print(type(image))
#     image = torch.from_numpy(image).to(device)
#     model = EfficientNet_Facial(2)
#     print(model)
#     output = model(image)
#     print(np.shape(output))
#     print(output)
# #     for name, param in model.named_parameters():
# #         print(name, param.size())
# #     model.backbone._fc.bias.requires_grad = False
# #     model.backbone._fc.weight.requires_grad = False
#     for param in model.parameters():
#         param.requires_grad = False
#     model.backbone._fc.bias.requires_grad = True
#     model.backbone._fc.weight.requires_grad = True
    # image = np.zeros((10,3, 224,224), dtype = np.float32)
    # landmark = np.zeros((10,3, 224,224), dtype = np.float32) + 255
    # image = torch.from_numpy(image)
    # model = Double_Net(2)
    # # model = model(include_top = False)
    # print(model)
    # feature = model(image, landmark)



    # out = model(image)
    print(np.shape(feature))

#     count  = 0
#     for child in model.children():
#         for layer in child:
#             print(layer)
#     for param in model.parameters():
# #         param.requires_grad = False
#         print(param)
    