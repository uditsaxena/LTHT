import torch
import torch.nn as nn
import torch.nn.functional as func

class fc1(nn.Module):

    def __init__(self, num_classes=10):
        super(fc1, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc1_bn = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 100)
        self.fc2_bn = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, num_classes)

        self.param_info = [{'layer_type':'Linear', 'name': 'Linear1'},
                            {'layer_type':'Linear', 'name': 'Linear2'},
                            {'layer_type':'Linear', 'name': 'Linear3'}]


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = func.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
