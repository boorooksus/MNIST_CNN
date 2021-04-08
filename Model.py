import torch
from torch.nn import Sequential,Linear,Sigmoid

class Model(torch.nn.Module):

  def __init__(self):
    # torch.nn.Module을 상속
    super(Model, self).__init__()

    # convolution layers
    conv1 = torch.nn.Conv2d(1, 16, 3, 1) 
    pool1 = torch.nn.MaxPool2d(2) 
    conv2 = torch.nn.Conv2d(16, 32, 3, 1) 
    pool2 = torch.nn.MaxPool2d(2)

    self.conv_layers = Sequential(
              conv1,
              torch.nn.ReLU(),
              pool1,
              conv2,
              torch.nn.ReLU(),
              pool2
              ).to('cuda')

    # fully connected layers
    fc1 = torch.nn.Linear(32 * 5 * 5, 120)
    fc2 = torch.nn.Linear(120, 32)
    fc3 = torch.nn.Linear(32, 10)        

    self.fc_layers = Sequential(
              fc1,
              torch.nn.ReLU(),
              fc2,
              torch.nn.ReLU(),
              fc3
          ).to('cuda')

  def forward(self, x):
    x = self.conv_layers(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layers(x)
    return x
