import math
from torch import nn
import torch

# 时间步编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=256):
        """
        restore dims
        :param dim:time emb dim
        """
        super().__init__()
        self.dim = dim
    def forward(self,
                t_idx:torch.Tensor):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t_idx.device) * -emb)
        x = t_idx[:, None].float() * emb[None, :]
        return torch.cat([x.sin(), x.cos()], dim=-1)

class BottleneckFiLM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        planes: base channels (like 64,128,256,...)
        out_channels after expansion = planes * expansion
        """
        super().__init__()
        out_channels = planes * self.expansion

        # conv1 1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # conv2 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # conv3 1x1
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, film_params=None):
        """
        film_params: None or tuple (scale, shift)
           scale, shift: each (B, C) or (B, C, 1, 1) where C = out_channels
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # apply FiLM if provided
        if film_params is not None:
            scale, shift = film_params  # expected shapes: (B, C) or (B,C,1,1)
            # ensure shape (B,C,1,1) for broadcasting
            if scale.dim() == 2:
                scale = scale.unsqueeze(-1).unsqueeze(-1)
            if shift.dim() == 2:
                shift = shift.unsqueeze(-1).unsqueeze(-1)
            out = out * (1.0 + scale) + shift

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet50FiLM(nn.Module):
    def __init__(self, num_classes, time_embed_dim=256, zero_init_residual=False):
        """
        num_classes: classifier classes
        time_embed_dim: dimension of time embedding (sinusoidal -> MLP -> time_emb)
        """
        super().__init__()
        self.inplanes = 64
        self.time_embed_dim = time_embed_dim

        # initial stem
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # prepare container for film projection layers (one per block)
        self.film_projs = nn.ModuleList()

        # layers: ResNet50 config [3,4,6,3]
        self.layer1 = self._make_layer(planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        # pooling + head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckFiLM.expansion, num_classes)

        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
            nn.SiLU()
        )

        # final init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # optionally zero-initialize the last BN in each residual branch, as described in ResNet paper
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckFiLM):
                    nn.init.constant_(m.bn3.weight, 0.0)

    def _make_layer(self, planes, blocks, stride=1):
        """
        Create a stage with 'blocks' BottleneckFiLM blocks.
        While creating each block, also create a film_proj Linear layer and append to self.film_projs.
        film_proj maps time_emb -> 2 * (planes * expansion) (scale + shift)
        """
        downsample = None
        out_channels = planes * BottleneckFiLM.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # first block (may downsample)
        b = BottleneckFiLM(self.inplanes, planes, stride=stride, downsample=downsample)
        layers.append(b)
        # create film proj for this block
        self.film_projs.append(nn.Linear(self.time_embed_dim, 2 * out_channels))
        self.inplanes = out_channels

        # remaining blocks
        for _ in range(1, blocks):
            b = BottleneckFiLM(self.inplanes, planes)
            layers.append(b)
            self.film_projs.append(nn.Linear(self.time_embed_dim, 2 * out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, t):
        """
        x: (B,1,H,W) noisy images x_t
        t: (B,) timesteps (int or float)
        returns logits (B, num_classes)
        """
        # get timestep_embedding
        get_te = SinusoidalPosEmb()

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # time embedding
        # compute sinusoidal then MLP -> time_emb (B, time_embed_dim)
        te = get_te(t)  # (B, time_embed_dim)
        time_emb = self.time_mlp(te)  # (B, time_embed_dim)

        # iterate through layers & blocks, consuming film_projs in order
        film_idx = 0

        # helper to run a sequential layer with film
        def run_layer(layer_seq, x, film_idx):
            for block in layer_seq:
                # get film proj corresponding to this block
                film_proj = self.film_projs[film_idx]  # Linear(time_emb -> 2*C)
                film_raw = film_proj(time_emb)  # (B, 2*C)
                C = film_raw.shape[1] // 2
                scale = film_raw[:, :C]
                shift = film_raw[:, C:]
                x = block(x, film_params=(scale, shift))
                film_idx += 1
            return x, film_idx

        x, film_idx = run_layer(self.layer1, x, film_idx)
        x, film_idx = run_layer(self.layer2, x, film_idx)
        x, film_idx = run_layer(self.layer3, x, film_idx)
        x, film_idx = run_layer(self.layer4, x, film_idx)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

if __name__ == "__main__":
    import torch
    T = 1_000
    x = torch.randn(size=[4, 1, 224, 224], dtype=torch.float32)
    t_idx = torch.randint(0, T, size=[x.shape[0],])
    model = ResNet50FiLM(num_classes=3)
    print(model.avgpool)
    print(model(x, t_idx).shape)
