import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding': 1, 'name':'Conv1'},
                            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'Conv2'},
                            {'layer_type':'MaxPool2d', 'kernel_size':(2,2), 'stride':2, 'padding':0, 'name':'MaxPool1'},
                            {'layer_type':'Linear', 'name': 'Linear1'},
                            {'layer_type':'Linear', 'name': 'Linear2'},
                            {'layer_type':'Linear', 'name': 'Linear3'}]


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LeNet5_nmp(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_nmp, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Flatten(),
            nn.Conv2d(64, 64, kernel_size=(4,4), stride=2, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding': 1, 'name':'Conv1'},
                            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'Conv2'},
                            {'layer_type':'Conv2d', 'kernel_size':(4,4), 'stride':2, 'padding':1, 'name':'Conv3'},
                            {'layer_type':'Linear', 'name': 'Linear1'},
                            {'layer_type':'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}]


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
