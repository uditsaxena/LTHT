import torch.nn as nn
import torch.nn.functional as func


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 0, 'name': 'Conv1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 0, 'name': 'Conv2'},
                           {'layer_type': 'Linear', 'name': 'Linear1'},
                           {'layer_type': 'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}]

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_nmp(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_nmp, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(9216, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size':(5,5), 'stride':1, 'padding': 0, 'name':'Conv1'},
                            {'layer_type': 'Conv2d', 'kernel_size':(5,5), 'stride':1, 'padding':0, 'name':'Conv2'},
                            {'layer_type':'Linear', 'name': 'Linear1'},
                            {'layer_type':'Linear', 'name': 'Linear2'},
                            {'layer_type':'Linear', 'name': 'Linear3'}]

    def forward(self, x):
        x = func.relu(self.conv1(x))
        # x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        # x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x
