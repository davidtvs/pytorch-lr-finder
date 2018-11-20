import torch.nn as nn


__all__ = ["Cifar10ResNet", "resnet20", "resnet32", "resnet44", "resnet56", "resnet101"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Cifar10ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, ch_width=2):
        super(Cifar10ResNet, self).__init__()
        width = [16, 16 * ch_width, 16 * ch_width * ch_width]
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            3, width[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width[0], layers[0])
        self.layer2 = self._make_layer(block, width[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, width[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(num_classes=10, ch_width=2):
    """Constructs a ResNet-20 model.
    """
    return Cifar10ResNet(
        BasicBlock, [3, 3, 3], num_classes=num_classes, ch_width=ch_width
    )


def resnet32(num_classes=10, ch_width=2):
    """Constructs a ResNet-32 model.
    """
    return Cifar10ResNet(
        BasicBlock, [5, 5, 5], num_classes=num_classes, ch_width=ch_width
    )


def resnet44(num_classes=10, ch_width=2):
    """Constructs a ResNet-44 model.
    """
    return Cifar10ResNet(
        BasicBlock, [7, 7, 7], num_classes=num_classes, ch_width=ch_width
    )


def resnet56(num_classes=10, ch_width=2):
    """Constructs a ResNet-56 model.
    """
    return Cifar10ResNet(
        BasicBlock, [9, 9, 9], num_classes=num_classes, ch_width=ch_width
    )


def resnet101(num_classes=10, ch_width=2):
    """Constructs a ResNet-101 model.
    """
    return Cifar10ResNet(
        BasicBlock, [18, 18, 18], num_classes=num_classes, ch_width=ch_width
    )
