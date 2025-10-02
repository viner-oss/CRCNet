import torch
import torch.nn as nn
from typing import List, Union, Dict

_cfgs: Dict[str, List[Union[str,int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],            # VGG11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # VGG16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] # VGG19
}

def make_layers(cfg: List[Union[str,int]], batch_norm: bool=False, in_channels: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self,
                 features: nn.Sequential,
                 num_classes: int = 1000,
                 init_weights: bool = True,
                 classifier_dropout: float = 0.5,
                 avgpool_output_size: int = 7):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((avgpool_output_size, avgpool_output_size))
        in_features = 512 * (avgpool_output_size ** 2)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(classifier_dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(classifier_dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def _vgg(cfg_key: str, batch_norm: bool, in_channels: int = 3, **kwargs) -> VGG:
    cfg = _cfgs[cfg_key]
    features = make_layers(cfg, batch_norm=batch_norm, in_channels=in_channels)
    return VGG(features, **kwargs)

def vgg11(in_channels=3, num_classes=1000, batch_norm=False, **kwargs): return _vgg('A', batch_norm=batch_norm, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg11_bn(in_channels=3, num_classes=1000, **kwargs): return _vgg('A', batch_norm=True, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg13(in_channels=3, num_classes=1000, batch_norm=False, **kwargs): return _vgg('B', batch_norm=batch_norm, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg13_bn(in_channels=3, num_classes=1000, **kwargs): return _vgg('B', batch_norm=True, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg16(in_channels=3, num_classes=1000, batch_norm=False, **kwargs): return _vgg('D', batch_norm=batch_norm, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg16_bn(in_channels=3, num_classes=1000, **kwargs): return _vgg('D', batch_norm=True, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg19(in_channels=3, num_classes=1000, batch_norm=False, **kwargs): return _vgg('E', batch_norm=batch_norm, in_channels=in_channels, num_classes=num_classes, **kwargs)
def vgg19_bn(in_channels=3, num_classes=1000, **kwargs): return _vgg('E', batch_norm=True, in_channels=in_channels, num_classes=num_classes, **kwargs)

if __name__ == "__main__":
    model = vgg16_bn(in_channels=1, num_classes=3, avgpool_output_size=7)
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    print("Forward (1x224x224) ->", out.shape)  # (2,3)

