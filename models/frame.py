from collections import OrderedDict

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
from models.pooling import L2N, GeM, RMAC
from models.summary import summary

FRAME_MODELS = ['MobileNet_AVG', 'Resnet50_RMAC']


class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()


class MobileNet_AVG(BaseModel):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = nn.Sequential(OrderedDict(models.mobilenet_v2(pretrained=True).features.named_children()))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class Resnet50_RMAC(BaseModel):
    def __init__(self):
        super(Resnet50_RMAC, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x

class Resnet50(BaseModel):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.resnet_layers = dict(models.resnet50(pretrained=True).named_children())
        self.feat_ext = nn.Sequential(OrderedDict(
            [(key, self.resnet_layers[key])
             for key in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']]))

    def forward(self,x):
        x = self.feat_ext(x)
        return x


if __name__ == '__main__':
    model = Resnet50()
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model.__repr__())
    input = torch.zeros((16, 3, 224, 224))
    output = model(input)
    import pdb;pdb.set_trace()
    print(model)
