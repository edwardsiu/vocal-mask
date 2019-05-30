import torch
from torch import nn
import torch.nn.functional as F


def calc_padding(kernel):
    return list((i//2 for i in kernel))

class PreActResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3,3)):
        super().__init__()
        if in_planes == out_planes:
            stride = 1
            self.shortcut = None
        else:
            stride = 2
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        padding = calc_padding(kernel_size)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity

class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, activation, kernel, padding, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, stride=stride, groups=groups, bias=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, activation, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, groups=groups, bias=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class FC(nn.Module):
    def __init__(self, indims, outdims, activation):
        super().__init__()
        self.fc = nn.Linear(indims, outdims)
        self.bn = nn.BatchNorm1d(outdims)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.fc(x)))

class DepthConv(nn.Module):
    def __init__(self, indims, outdims, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(indims, indims, kernel_size=3, padding=1, stride=stride, groups=indims, bias=False)
        self.bn1 = nn.BatchNorm2d(indims)
        self.conv2 = nn.Conv2d(indims, outdims, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdims)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

