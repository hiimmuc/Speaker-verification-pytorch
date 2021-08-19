# import torch
import torch.nn as nn


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_shape, output_shape, stride=1, down_sample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, output_shape, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_shape)
        self.conv2 = nn.Conv2d(output_shape, output_shape, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_shape)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(output_shape, reduction)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_shape, output_shape, stride=1, down_sample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, output_shape, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_shape)
        self.conv2 = nn.Conv2d(output_shape, output_shape, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_shape)
        self.conv3 = nn.Conv2d(output_shape, output_shape * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_shape * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(output_shape * 4, reduction)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
