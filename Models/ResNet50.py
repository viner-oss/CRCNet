import torch
import torch.nn as nn
from typing import Union, Callable, Optional
from Models.factory import *


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm="bn",
                 act="relu",
                 num_groups=32,
                 use_bias=False):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, bias=use_bias)
        self.bn1 = get_norm(norm, planes, num_groups=num_groups)
        self.conv2 = conv3x3(planes, planes, stride, bias=use_bias)
        self.bn2 = get_norm(norm, planes, num_groups=num_groups)
        self.conv3 = conv1x1(planes, planes * self.expansion, bias=use_bias)
        self.bn3 = get_norm(norm, planes * self.expansion, num_groups=num_groups)

        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class ResNet50(nn.Module):
    def __init__(
        self,
        init_chs: int = 1,
        out_chs: int = 1,
        num_classes: int = 3,
        base_chs: int = 64,
        depth: int = 4,

        # alternative
        norm: Optional[str] = "bn",
        act: Optional[str] = "relu",
        dropout: float = 0.0,
        use_bias: Optional[bool] = False,
        num_groups: int = 32,
        **kwargs
    ):
        super().__init__()

        # ResNet50 block setting
        layers = [3, 4, 6, 3]

        self.inplanes = base_chs
        self.conv1 = nn.Conv2d(init_chs, base_chs, kernel_size=7,
                               stride=2, padding=3, bias=use_bias)
        self.bn1 = get_norm(norm, base_chs, num_groups=num_groups)
        self.act_fn = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_chs, layers[0], norm, act, num_groups, use_bias)
        self.layer2 = self._make_layer(base_chs * 2, layers[1], norm, act, num_groups, use_bias, stride=2)
        self.layer3 = self._make_layer(base_chs * 4, layers[2], norm, act, num_groups, use_bias, stride=2)
        self.layer4 = self._make_layer(base_chs * 8, layers[3], norm, act, num_groups, use_bias, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_chs * 8 * Bottleneck.expansion, num_classes)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _make_layer(self, planes, blocks, norm, act, num_groups, use_bias, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride, bias=use_bias),
                get_norm(norm, planes * Bottleneck.expansion, num_groups=num_groups)
            )

        layers = []
        layers.append(
            Bottleneck(self.inplanes, planes, stride, downsample,
                       norm=norm, act=act, num_groups=num_groups, use_bias=use_bias)
        )
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(self.inplanes, planes,
                           norm=norm, act=act, num_groups=num_groups, use_bias=use_bias)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    x = torch.randn(size=[4, 1, 224, 224])
    model = ResNet50()
    print(model(x).shape)
    print(model)