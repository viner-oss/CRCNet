from Models.factory import *
class DepthWiseSeparableConv(nn.Module):
    def __init__(self,
                 in_chs: int,
                 out_chs: int,
                 stride: int,
                 norm: Optional[str] = 'bn',
                 act: Optional[str] = 'relu6',
                 num_groups: int = 32):
        super(DepthWiseSeparableConv, self).__init__()
        self.block = nn.Sequential(
            # DepthWise
            nn.Conv2d(in_chs, in_chs, 3, stride, 1, groups=in_chs, bias=False),
            get_norm(norm, in_chs, num_groups),
            get_activation(act),

            # PointWise
            nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
            get_norm(norm, out_chs, num_groups),
            get_activation(act)
        )

    def forward(self,
                x: torch.Tensor):
        return self.block(x)

class MobileNetV1(nn.Module):
    def __init__(self,
                 init_chs: int = 1,
                 out_chs: int = 1,
                 num_classes: int = 3,
                 width_mult: float = 1.0,

                 # alternative
                 norm: Optional[str] = 'bn',
                 act: Optional[str] = 'relu6',
                 dropout: float = 0.2,
                 use_bias: Optional[bool] = False,
                 num_groups: int = 32,
                 use_extract: bool = False,
                 **kwargs):
        super(MobileNetV1, self).__init__()
        self.use_extract = use_extract

        def c(chs: int) -> int:
            return max(8, int(chs * width_mult))

        # layer1
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(init_chs, c(32), 3, stride=2, padding=1, bias=use_bias),
            get_norm(norm, c(32), num_groups),
            get_activation(act)
        ))

        # layer2
        layers.append(DepthWiseSeparableConv(c(32), c(64), stride=1, norm=norm, act=act, num_groups=num_groups))
        layers.append(DepthWiseSeparableConv(c(64), c(128), stride=2, norm=norm, act=act, num_groups=num_groups))
        layers.append(DepthWiseSeparableConv(c(128), c(128), stride=1, norm=norm, act=act, num_groups=num_groups))
        layers.append(DepthWiseSeparableConv(c(128), c(256), stride=2, norm=norm, act=act, num_groups=num_groups))
        layers.append(DepthWiseSeparableConv(c(256), c(256), stride=1, norm=norm, act=act, num_groups=num_groups))
        layers.append(DepthWiseSeparableConv(c(256), c(512), stride=2, norm=norm, act=act, num_groups=num_groups))

        # layer3
        for _ in range(5):
            layers.append(DepthWiseSeparableConv(c(512), c(512), stride=1, norm=norm, act=act, num_groups=num_groups))

        # layer4
        layers.append(DepthWiseSeparableConv(c(512), c(1024), stride=2, norm=norm, act=act, num_groups=num_groups))
        layers.append(DepthWiseSeparableConv(c(1024), c(1024), stride=1, norm=norm, act=act, num_groups=num_groups))

        self.feats = nn.Sequential(*layers)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity
        self.fc = nn.Linear(c(1024), num_classes)

    def forward(self,
                x: torch.Tensor):
        x = self.feats(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_extract:
            return x

        if not self.use_extract:
            x = self.dropout(x)
            out = self.fc(x)
            return out

if __name__ == '__main__':
    # x = torch.randn(size=[4, 1, 224, 224])
    # model = MobileNetV1()
    # print(model(x).shape)
    # print(model)
    pass






