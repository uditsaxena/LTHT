import torch
import torch.nn as nn
import torch.nn.functional as func


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2),
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
        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 2, 'name': 'Conv1'},
                           {'layer_type': 'MaxPool2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 0,
                            'name': 'MaxPool1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 2, 'name': 'Conv2'},
                           {'layer_type': 'MaxPool2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 0,
                            'name': 'MaxPool2'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv3'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv4'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv5'},
                           {'layer_type': 'MaxPool2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 0,
                            'name': 'MaxPool3'},
                           {'layer_type': 'Linear', 'name': 'Linear1'},
                           {'layer_type': 'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}, ]

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class AlexNet_bn(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_bn, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(192)

        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(256, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096)

        self.fc3 = nn.Linear(4096, num_classes)

        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 2, 'name': 'Conv1'},
                           {'layer_type': 'MaxPool2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 0,
                            'name': 'MaxPool1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 2, 'name': 'Conv2'},
                           {'layer_type': 'MaxPool2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 0,
                            'name': 'MaxPool2'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv3'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv4'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv5'},
                           {'layer_type': 'MaxPool2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 0,
                            'name': 'MaxPool3'},
                           {'layer_type': 'Linear', 'name': 'Linear1'},
                           {'layer_type': 'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}, ]

    def forward(self, x):
        x = func.relu(self.conv1_bn(self.conv1(x)))
        x = self.mp1(x)
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = self.mp2(x)
        x = func.relu(self.conv3_bn(self.conv3(x)))
        x = func.relu(self.conv4_bn(self.conv4(x)))
        x = func.relu(self.conv5_bn(self.conv5(x)))
        x = self.mp3(x)
        x = torch.flatten(x, 1)
        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = func.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x



class AlexNet_nmp(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_nmp, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 192, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 2, 'name': 'Conv1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (4, 4), 'stride': 2, 'padding': 1, 'name': 'Conv2'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 2, 'name': 'Conv3'},
                           {'layer_type': 'Conv2d', 'kernel_size': (4, 4), 'stride': 2, 'padding': 1, 'name': 'Conv4'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv5'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv6'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv7'},
                           {'layer_type': 'Conv2d', 'kernel_size': (4, 4), 'stride': 2, 'padding': 1, 'name': 'Conv8'},
                           {'layer_type': 'Linear', 'name': 'Linear1'},
                           {'layer_type': 'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet_nmp_bn(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_nmp_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3_bn = nn.BatchNorm2d(192)

        self.conv4 = nn.Conv2d(192, 192, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(192)

        self.conv5 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(384)

        self.conv6 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv8_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096)

        self.fc3 = nn.Linear(4096, num_classes)

        self.param_info = [{'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 2, 'padding': 2, 'name': 'Conv1'},
                           {'layer_type': 'Conv2d', 'kernel_size': (4, 4), 'stride': 2, 'padding': 1, 'name': 'Conv2'},
                           {'layer_type': 'Conv2d', 'kernel_size': (5, 5), 'stride': 1, 'padding': 2, 'name': 'Conv3'},
                           {'layer_type': 'Conv2d', 'kernel_size': (4, 4), 'stride': 2, 'padding': 1, 'name': 'Conv4'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv5'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv6'},
                           {'layer_type': 'Conv2d', 'kernel_size': (3, 3), 'stride': 1, 'padding': 1, 'name': 'Conv7'},
                           {'layer_type': 'Conv2d', 'kernel_size': (4, 4), 'stride': 2, 'padding': 1, 'name': 'Conv8'},
                           {'layer_type': 'Linear', 'name': 'Linear1'},
                           {'layer_type': 'Linear', 'name': 'Linear2'},
                           {'layer_type': 'Linear', 'name': 'Linear3'}]

    def forward(self, x):

        x = func.relu(self.conv1_bn(self.conv1(x)))
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = func.relu(self.conv3_bn(self.conv3(x)))
        x = func.relu(self.conv4_bn(self.conv4(x)))
        x = func.relu(self.conv5_bn(self.conv5(x)))
        x = func.relu(self.conv6_bn(self.conv6(x)))
        x = func.relu(self.conv7_bn(self.conv7(x)))
        x = func.relu(self.conv8_bn(self.conv8(x)))

        x = torch.flatten(x, 1)

        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = func.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x
