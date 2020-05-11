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
                            {'layer_type':'MaxPool2d', 'kernel_size':(2,2), 'stride':2, 'padding':0, 'name':'MaxPool1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 0, 'name': 'Conv2'},
                           {'layer_type':'MaxPool2d', 'kernel_size':(2,2), 'stride':2, 'padding':0, 'name':'MaxPool2'},
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


class LeNet5_bn(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)
        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 0, 'name': 'Conv1'},
                            {'layer_type':'MaxPool2d', 'kernel_size':(2,2), 'stride':2, 'padding':0, 'name':'MaxPool1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 0, 'name': 'Conv2'},
                           {'layer_type':'MaxPool2d', 'kernel_size':(2,2), 'stride':2, 'padding':0, 'name':'MaxPool2'},
                           {'layer_type': 'Linear', 'name': 'Linear1'},
                           {'layer_type': 'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}]

    def forward(self, x):
        x = func.relu(self.conv1_bn(self.conv1(x)))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = func.relu(self.fc2_bn(self.fc2(x)))
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


class LeNet5_nmp_bn(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_nmp_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(9216, 120, bias=False)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes, bias=False)

        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size':(5,5), 'stride':1, 'padding': 0, 'name':'Conv1'},
                            {'layer_type': 'Conv2d', 'kernel_size':(5,5), 'stride':1, 'padding':0, 'name':'Conv2'},
                            {'layer_type':'Linear', 'name': 'Linear1'},
                            {'layer_type':'Linear', 'name': 'Linear2'},
                            {'layer_type':'Linear', 'name': 'Linear3'}]

    def forward(self, x):
        x = func.relu(self.conv1_bn(self.conv1(x)))
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = func.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
