import math
import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=256):
        """
        restore dims
        :param dim:time emb dim
        """
        super().__init__()
        self.dim = dim

    def forward(self,
                t_idx: torch.Tensor):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t_idx.device) * -emb)
        x = t_idx[:, None].float() * emb[None, :]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 time_emb_dim: int = 256,
                 dropout: float = 0.1):
        super(ResBlock, self).__init__()
        self.feat_extract1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,
                      out_channels=out_ch,
                      kernel_size=3,
                      padding=1),
            nn.GroupNorm(num_groups=8,
                         num_channels=out_ch),
            nn.SiLU()
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dim,
                      out_features=out_ch)
        )
        self.feat_extract2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch,
                      out_channels=out_ch,
                      kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8,
                         num_channels=out_ch),
            nn.Dropout(p=dropout)
        )
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_channels=in_ch,
                                  out_channels=out_ch,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1) if in_ch != out_ch else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor):
        h = self.feat_extract1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.feat_extract2(h)
        return self.act(h + self.res_conv(x))


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 num_head: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.channels = channels
        assert channels % num_head == 0, 'channels must be divisible by num_head'
        self.head_dim = channels // num_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(in_channels=channels, out_channels=channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self,
                x: torch.Tensor):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape([B, self.num_head, self.head_dim, H * W])
        k = k.reshape([B, self.num_head, self.head_dim, H * W])
        v = v.reshape([B, self.num_head, self.head_dim, H * W])

        attn = torch.einsum("bnqd, bnkd->bnqk", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bnqk, bnvd->bnqd", attn, v)
        out = out.reshape([B, C, H, W])
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 num_resblock: int):
        super(EncoderBlock, self).__init__()
        self.resblock = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_resblock)
        ])
        self.down = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor):
        for rb in self.resblock:
            x = rb(x, t_emb)
        out = self.down(x)
        return out


class BottleNeck(nn.Module):
    def __init__(self,
                 channels: int,
                 time_emb_dim: int,
                 num_head: int):
        super(BottleNeck, self).__init__()
        self.res1 = ResBlock(channels, channels, time_emb_dim)
        self.attn = MultiHeadAttention(channels, num_head)
        self.res2 = ResBlock(channels, channels, time_emb_dim)

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        out = self.res2(x, t_emb)
        return out


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 num_resblock: int):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.resblock = nn.ModuleList([
            ResBlock(in_channels + skip_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_resblock)
        ])

    def forward(self,
                x: torch.Tensor,
                skip: torch.Tensor,
                t_emb: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for rb in self.resblock:
            x = rb(x, t_emb)
        return x


class Unet(nn.Module):
    """
    stem   tensor[3, 224, 224]--->[64, 224, 224]    skip1
    e1     tensor[64, 224, 224]--->[128, 112, 112]  skip2
    e2     tensor[128, 112, 112]--->[256, 56, 56]   skip3
    e3     tensor[256, 56, 56]--->[512, 28, 28]     skip4
    e4     tensor[512, 28, 28]--->[1024, 14, 14]
    """

    def __init__(self, t_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(),
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.GroupNorm(32, 64),
            nn.SiLU()
        )

        self.encoders = nn.ModuleList([
            EncoderBlock(64, 128, 256, 2),
            EncoderBlock(128, 256, 256, 2),
            EncoderBlock(256, 512, 256, 2),
            EncoderBlock(512, 1024, 256, 2)
        ])

        self.bots = nn.ModuleList([
            BottleNeck(1024, 256, 8)
        ])

        self.decoders = nn.ModuleList([
            DecoderBlock(1024, 512, 512, 256, 2),
            DecoderBlock(512, 256, 256, 256, 2),
            DecoderBlock(256, 128, 128, 256, 2),
            DecoderBlock(128, 64, 64, 256, 2)
        ])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self,
                x: torch.Tensor,
                t_idx: torch.Tensor):
        skip_images = []

        t_emb = self.time_mlp(t_idx)
        x = self.stem(x)
        skip_images.append(x.detach())

        for i, block in enumerate(self.encoders):
            x = block(x, t_emb)
            if i != (len(self.encoders) - 1):
                skip_images.append(x.detach())

        for block in self.bots:
            x = block(x, t_emb)

        for block in self.decoders:
            skip = skip_images.pop()
            x = block(x, skip, t_emb)
            del skip

        out = self.out_conv(x)

        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = 1_000

    model = Unet().to(device)
    x = torch.randn(size=[4, 1, 224, 224], dtype=torch.float32)
    t_idx = torch.randint(0, T, size=(x.shape[0],))
    print(model(x, t_idx).shape)
    print(model)
