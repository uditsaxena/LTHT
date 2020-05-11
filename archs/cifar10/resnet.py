'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion*planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.param_info = param_info = [{'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding': 1, 'name':'conv1', 'param_idx':0},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer1.0.conv1', 'param_idx':1},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer1.0.conv2', 'param_idx':2},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer1.0.shortcut', 'connects':[1,4], 'param_idx':3},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer1.1.conv1', 'param_idx':5},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer1.1.conv2', 'param_idx':6},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer1.1.shortcut', 'connects':[4,7], 'param_idx':7},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':2, 'padding':1, 'name':'layer2.0.conv1', 'param_idx':9},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer2.0.conv2', 'param_idx':10},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer2.0.shortcut', 'connects':[7,10], 'param_idx':11},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer2.1.conv1', 'param_idx':13},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer2.1.conv2', 'param_idx':14},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer2.1.shortcut', 'connects':[10,13], 'param_idx':15},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':2, 'padding':1, 'name':'layer3.0.conv1', 'param_idx':17},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer3.0.conv2', 'param_idx':18},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer3.0.shortcut', 'connects':[13,16], 'param_idx':19},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer3.1.conv1', 'param_idx':21},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer3.1.conv2', 'param_idx':22},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer3.1.shortcut', 'connects':[16,19], 'param_idx':23},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':2, 'padding':1, 'name':'layer4.0.conv1', 'param_idx':25},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer4.0.conv2', 'param_idx':26},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer4.0.shortcut', 'connects':[19,22], 'param_idx':27},

            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer4.1.conv1', 'param_idx':29},
            {'layer_type': 'Conv2d', 'kernel_size':(3,3), 'stride':1, 'padding':1, 'name':'layer4.1.conv2', 'param_idx':30},
            {'layer_type': 'Shortcut', 'kernel_size': (1,1), 'stride':1, 'padding':0, 'name':'layer4.1.shortcut', 'connects':[22,25], 'param_idx':31},

            {'layer_type':'MaxPool2d', 'kernel_size':(4,4), 'stride':4, 'padding':0, 'name':'MaxPool'},
            {'layer_type':'Linear', 'name': 'Linear1', 'param_idx':33}]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
