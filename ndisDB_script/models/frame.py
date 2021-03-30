from collections import OrderedDict

from torchvision import models
import torch.nn as nn
import torch
from ndisDB_script.models.pooling import L2N, RMAC
from ndisDB_script.models.summary import summary

FRAME_MODELS = ['MobileNet_AVG', 'Resnet50_RMAC','Resnet50_avgpool_FC3','Resnet50_ccpool_FC3']


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
        self.base = nn.Sequential(OrderedDict(
            [(key, self.resnet_layers[key])
             for key in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3','layer4']]))
    def forward(self,x):
        x = self.base(x)

        return x

class Resnet50_avgpool_FC3(BaseModel):
    def __init__(self,D_in=2048,hidden_layer_sizes=[2000,1000,512]):
        super(Resnet50_avgpool_FC3, self).__init__()
        self.Resnet50 = Resnet50()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.D_in = D_in
        self.hidden_size = hidden_layer_sizes
        self.D_out = hidden_layer_sizes[-1]

        self.fc1 = torch.nn.Sequential(
            nn.Linear(self.D_in, self.hidden_size[0]),
            nn.Tanh() # tanh
        )
        self.fc2 = torch.nn.Sequential(
            nn.Linear(self.hidden_size[0],self.hidden_size[1]),
            nn.Tanh()# tanh
        )
        self.fc3 = torch.nn.Sequential(
            nn.Linear(self.hidden_size[1],self.D_out),
            nn.Tanh()# tanh
        )
        self.norm = L2N()


    def forward(self,x):
        x = self.Resnet50(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.norm(x)
        return x


class Resnet50_ccpool_FC3(BaseModel):
    def __init__(self,D_in=3136,hidden_layer_sizes=[2000,1000,512]):
        super(Resnet50_ccpool_FC3, self).__init__()
        self.Resnet50 = Resnet50()
        self.pool = torch.nn.AvgPool3d(kernel_size=(32, 1, 1), stride=(32, 1, 1))
        self.D_in = D_in
        self.hidden_size = hidden_layer_sizes
        self.D_out = hidden_layer_sizes[-1]

        self.fc1 = torch.nn.Sequential(
            nn.Linear(self.D_in, self.hidden_size[0]),
            nn.Tanh()  # tanh
        )
        self.fc2 = torch.nn.Sequential(
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.Tanh()  # tanh
        )
        self.fc3 = torch.nn.Sequential(
            nn.Linear(self.hidden_size[1], self.D_out),
            nn.Tanh()  # tanh
        )
        self.norm = L2N()

    def forward(self,x):
        x = self.Resnet50(x)
        x = self.pool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.norm(x)
        return x

class Resnet34_ccpool(BaseModel):
    def __init__(self):
        super(Resnet34_ccpool,self).__init__()
        self.resnet_layer = models.resnet34(pretrained=True)
        self.modules = list(self.resnet_layer.children())
        self.modules = self.modules[:-2]
        self.resnet_layer = nn.Sequential(*self.modules)
        self.resnet_layer.add_module('1*1_averagepooling', nn.AvgPool3d(kernel_size=(8,1,1), stride=(8,1,1)))

        self.fc = nn.Linear(3136, 512)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = Resnet50_FC3()
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model.__repr__())
    input = torch.zeros((16, 3, 224, 224))
    output = model(input)
    # import pdb;pdb.set_trace()
    print(model)
