import torch
import torch.nn as nn


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':2, 'padding': 2, 'name':'Conv1'},
                            {'layer_type':'MaxPool2d', 'kernel_size':(3,3), 'stride':2, 'padding':0, 'name':'MaxPool1'},
                            {'layer_type': 'Conv2d', 'kernel_size':(5,5), 'stride':1, 'padding':2, 'name':'Conv2'},
                            {'layer_type':'MaxPool2d', 'kernel_size':(3,3), 'stride':2, 'padding':0, 'name':'MaxPool2'},
                            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'Conv3'},
                            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'Conv4'},
                            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'Conv5'},
                            {'layer_type':'MaxPool2d', 'kernel_size':(3,3), 'stride':2, 'padding':0, 'name':'MaxPool3'},
                            {'layer_type':'Linear', 'name': 'Linear1'},
                            {'layer_type':'Linear', 'name': 'Linear2'},
                            {'layer_type':'Linear', 'name': 'Linear3'},]

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
