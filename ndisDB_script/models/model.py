import torch.nn as nn
import torch
from ndisDB_script.models import L2N
from ndisDB_script.models.summary import summary

FRAME_MODELS = ['MobileNet_AVG', 'Resnet50_RMAC']


class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()

class DNN(BaseModel):
    def __init__(self,D_in,hidden_layer_sizes):
        super(DNN, self).__init__()
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
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = self.norm(x)
        return x


if __name__ == '__main__':
    model = DNN(D_in=2048, hidden_layer_sizes=[2000,1000,512])
    print(model.summary((16, 2048), device='cpu'))
    print(model.__repr__())
    input = torch.zeros((16, 2048))
    output = model(input)
    # import pdb;pdb.set_trace()
    # print(model)
