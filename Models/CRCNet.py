import math
from typing import Optional
from abc import abstractmethod
import numpy as np
from Models.ResNet50FiLM import *
from Models.MobileNetV1 import *
from .factory import *
import torch.nn.functional as F
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

class TimeRoIBlock(nn.Module):
    @abstractmethod
    def forward(self, x, t_emb, low_emb, high_emb):
        """
        Apply the module to `x` given `emb` timestep embeddings and roi_embeddings.
        """

class EmbSequential(nn.Sequential, TimeRoIBlock):

    def forward(
            self,
            x,
            t_emb=None,
            low_emb=None,
            high_emb=None
    ):
        for layer in self:
            if isinstance(layer, TimeRoIBlock):
                x = layer(x, t_emb, low_emb, high_emb)
            else:
                x = layer(x)
            if x is None:
                raise RuntimeError(f"Layer {layer.__class__.__name__} returned None")
        return x

class Upsample(nn.Module):
    """
    Originally ported from here, but simplify
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L81
    """
    def __init__(
            self,
            in_chs,
            use_conv,
            out_chs = None
    ):
        super(Upsample, self).__init__()
        self.in_chs = in_chs
        self.use_conv = use_conv
        self.out_chs = out_chs or in_chs
        if use_conv:
            self.conv = nn.Conv2d(self.in_chs, self.out_chs, 3, padding=1)

    def forward(
            self,
            x: torch.Tensor
    ):
        assert x.shape[1] == self.in_chs, "x.shape[1] must be equal to in_channels"
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    Originally ported from here, but simplify
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L81
    """
    def __init__(
            self,
            in_chs,
            use_conv,
            out_chs=None
    ):
        super(Downsample, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs or in_chs
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.in_chs, self.out_chs, 3, stride=stride, padding=1
            )
        else:
            assert self.in_chs == self.out_chs, "when try to use avg_pool, in_channels must be equal to out_channels"
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(
            self,
            x: torch.Tensor
    ):
        assert x.shape[1] == self.in_chs, "x.shape[1] must be equal to in_channels"
        return self.op(x)

class ResBlock(TimeRoIBlock):
    """
    Originally ported from here, but adapted to both t_emb and roi_emb case.
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L81
    """
    def __init__(
            self,
            in_chs,
            t_emb_dims,
            low_emb_dims,
            high_emb_dims,
            out_chs = None,
            norm: Optional[str] = 'gn',
            act: Optional[str] = 'silu',
            dropout: float = 0.0,
            use_high_emb = False,
            use_conv=False,
            use_scale_shift_norm=False,
            up=False,
            down=False,
    ):
        super(ResBlock, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs or in_chs
        self.t_emb_dims = t_emb_dims
        self.low_emb_dims = low_emb_dims
        self.high_emb_dims = high_emb_dims
        self.use_high_emb = use_high_emb
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            get_norm(norm, in_chs),
            get_activation(act),
            nn.Conv2d(in_chs, self.out_chs, 3, padding=1)
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_chs, False)
            self.x_upd = Upsample(in_chs, False)
        elif down:
            self.h_upd = Downsample(in_chs, False)
            self.x_upd = Downsample(in_chs, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.t_emb_layers = nn.Sequential(
            get_activation(act),
            nn.Linear(
                t_emb_dims,
                self.out_chs * 2 if use_scale_shift_norm else self.out_chs
            )
        )
        self.latent_emb_layers = nn.Sequential(
            get_activation(act),
            nn.Linear(
                low_emb_dims,
                self.out_chs * 2 if use_scale_shift_norm else self.out_chs
            )
        ) if not use_high_emb else (
            nn.Sequential(
            get_activation(act),
            nn.Linear(
                high_emb_dims,
                self.out_chs * 2 if use_scale_shift_norm else self.out_chs
            )
        ))

        self.out_layers = nn.Sequential(
            get_norm(norm, self.out_chs),
            get_activation(act),
            nn.Dropout(dropout),
            zero_module(
                nn.Conv2d(self.out_chs, self.out_chs, 3, padding=1)
            )
        )

        if self.out_chs == self.in_chs:
            self.res_conv = nn.Identity()
        elif use_conv:
            self.res_conv = nn.Conv2d(
                self.in_chs, self.out_chs, 3, padding=1
            )
        else:
            self.res_conv = nn.Conv2d(self.in_chs, self.out_chs, 1)

    def forward(
            self,
            x: torch.Tensor,
            t_emb: torch.Tensor,
            low_emb: torch.Tensor,
            high_emb: torch.Tensor
    ):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        t_emb_out = self.t_emb_layers(t_emb).type(h.dtype)
        latent_emb_out = self.latent_emb_layers(low_emb).type(h.dtype) if not self.use_high_emb else (
            self.latent_emb_layers(high_emb).type(h.dtype))

        while len(t_emb_out.shape) < len(h.shape):
            t_emb_out = t_emb_out[..., None]
        while len(latent_emb_out.shape) < len(h.shape):
            latent_emb_out = latent_emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, _ = torch.chunk(t_emb_out, 2, dim=1)
            _, shift = torch.chunk(latent_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + t_emb_out + latent_emb_out
            h = self.out_layers(h)
        return self.res_conv(x) + h

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        in_chs,
        num_heads=1,
        num_head_channels=-1,
        norm: Optional[str] = 'gn',
        use_new_attention_order=False,
    ):
        super().__init__()
        self.in_chs = in_chs
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                in_chs % num_head_channels == 0
            ), f"q,k,v channels {in_chs} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = in_chs // num_head_channels
        self.norm = get_norm(norm, in_chs)
        self.qkv = nn.Conv1d(in_chs, in_chs * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(in_chs, in_chs, 1))

    def forward(
            self,
            x: torch.Tensor):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class Unet(nn.Module):
    def __init__(
            self,
            image_size,
            num_classes,
            init_ch,
            base_chs,
            out_chs,
            low_in_chs,
            high_in_chs,
            num_res_block,
            attn_resolutions,
            dropout = 0.0,
            ch_mult=(1, 1, 2, 2, 4, 4),
            num_heads = 1,
            num_head_chs = -1,
            num_heads_upsample=-1,
            norm: Optional[str] = 'gn',
            act: Optional[str] = 'silu',
            conv_resample=True,
            use_fp16 = False,
            use_scale_shift_norm = True,
            use_resblock_updown = True,
            use_new_attn_order = False,
            **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.init_chs = init_ch
        self.base_chs = base_chs
        self.out_chs = out_chs
        self.num_res_block = num_res_block
        self.attention_resolutions = attn_resolutions
        self.dropout = dropout
        self.ch_mult = ch_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_chs
        self.dtype = torch.float16 if use_fp16 else torch.float32

        t_emb_dim = low_emb_dim = base_chs * 4
        high_emb_dim = base_chs * ch_mult[-1]
        self.time_mlp = nn.Sequential(
            nn.Linear(base_chs, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        self.low_mlp = nn.Sequential(
            nn.Linear(low_in_chs, low_emb_dim),
            nn.SiLU(),
            nn.Linear(low_emb_dim, low_emb_dim)
        )
        self.high_mlp = nn.Sequential(
            nn.Linear(high_in_chs, high_emb_dim),
            nn.SiLU(),
            nn.Linear(high_emb_dim, high_emb_dim)
        )


        ch = input_ch = int(ch_mult[0] * base_chs)
        self.input_blocks = nn.ModuleList(
            [
                EmbSequential(nn.Conv2d(init_ch, ch, 3, padding=1))
            ]
        )
        self._feat_size = ch
        input_block_chs = [ch]
        ds = 1
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res_block):
                layers = [
                    ResBlock(
                        in_chs=ch,
                        t_emb_dims=t_emb_dim,
                        low_emb_dims=low_emb_dim,
                        high_emb_dims=high_emb_dim,
                        out_chs=int(mult * base_chs),
                        dropout=dropout,
                        use_high_emb=False,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = int(mult * base_chs)
                if ds in attn_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_chs,
                            use_new_attention_order=use_new_attn_order
                        )
                    )
                self.input_blocks.append(EmbSequential(*layers))
                self._feat_size += ch
                input_block_chs.append(ch)
            if level != len(ch_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbSequential(
                        ResBlock(
                            in_chs=ch,
                            t_emb_dims=t_emb_dim,
                            low_emb_dims=low_emb_dim,
                            high_emb_dims=high_emb_dim,
                            out_chs=out_ch,
                            dropout=dropout,
                            use_high_emb=False,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if use_resblock_updown
                        else Downsample(
                            ch, conv_resample, out_chs=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chs.append(ch)
                ds *= 2
                self._feat_size += ch

        self.middle_block = EmbSequential(
            ResBlock(
                in_chs=ch,
                t_emb_dims=t_emb_dim,
                low_emb_dims=low_emb_dim,
                high_emb_dims=high_emb_dim,
                dropout=dropout,
                use_high_emb=True,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_chs,
                use_new_attention_order=use_new_attn_order,
            ),
            ResBlock(
                in_chs=ch,
                t_emb_dims=t_emb_dim,
                low_emb_dims=low_emb_dim,
                high_emb_dims=high_emb_dim,
                dropout=dropout,
                use_high_emb=True,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feat_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(num_res_block + 1):
                ich = input_block_chs.pop()
                layers = [
                    ResBlock(
                        in_chs=ch + ich,
                        t_emb_dims=t_emb_dim,
                        low_emb_dims=low_emb_dim,
                        high_emb_dims=high_emb_dim,
                        dropout=dropout,
                        out_chs=int(base_chs * mult),
                        use_high_emb=False,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(base_chs * mult)
                if ds in attn_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_chs,
                            use_new_attention_order=use_new_attn_order,
                        )
                    )
                if level and i == num_res_block:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            in_chs=ch,
                            t_emb_dims=t_emb_dim,
                            low_emb_dims=low_emb_dim,
                            high_emb_dims=high_emb_dim,
                            dropout=dropout,
                            out_chs=out_ch,
                            use_high_emb=False,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if use_resblock_updown
                        else Upsample(ch, conv_resample, out_chs=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbSequential(*layers))
                self._feat_size += ch

        self.out = nn.Sequential(
            get_norm(norm, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_chs, 3, padding=1)),
        )

        self.t_pos_emb = SinusoidalPosEmb(self.base_chs)

    def forward(
            self,
            x: torch.Tensor,
            t_idx: torch.Tensor,
            low_semantic: torch.Tensor,
            high_semantic: torch.Tensor
    ):
        """
        :param x: an [B x C x W x H] Tensor of input
        :param t_idx: an [B, ] tensor of time_step
        :param roi: an [B x C x ... x ...] of latent feats
        :return: an [B x C x W x H] Tensor of output
        """
        hs = []
        latent_feats = []

        
        t_eb = self.t_pos_emb(t_idx)
        if len(low_semantic.shape) == 2:
            low_eb = low_semantic
        else:
            low_eb = F.adaptive_avg_pool2d(low_semantic, 1).view(low_semantic.shape[0], low_semantic.shape[1])
        if len(high_semantic.shape) == 2:
            high_eb = high_semantic
        else:
            high_eb = F.adaptive_avg_pool2d(high_semantic, 1).view(high_semantic.shape[0], high_semantic.shape[1])

        t_emb = self.time_mlp(t_eb)
        low_emb = self.low_mlp(low_eb)
        high_emb = self.high_mlp(high_eb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, t_emb, low_emb, None)
            hs.append(h)
        h = self.middle_block(h, t_emb, None, high_emb)
        latent_feats.append(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, t_emb, low_emb, None)

            if h.shape == latent_feats[-1].shape:
                latent_feats.pop()
                latent_feats.append(h)
            elif h.shape[1] != latent_feats[-1].shape[1]:
                latent_feats.append(h)
            else:
                continue
        h = h.type(x.dtype)
        return self.out(h), latent_feats

# ======================================================================================================================
class SEBlock(nn.Module):
    def __init__(self,
                 channels:int,
                 reduction:int = 16,
                 act: Optional[str] = 'relu'):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            get_activation(act),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self,
                x:torch.Tensor):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DetectHead(nn.Module):
    def __init__(self,
                 base_chs,
                 ch_mult,
                 resolution,
                 n_head: int = 3,
                 num_classes: int = 3,
                 num_cbr: int = 5,
                 dropout: float = 0.0,
                 norm: Optional[str] = 'bn',
                 act: Optional[str] = 'relu'):
        super().__init__()
        self.n_head = n_head

        out_chs = max(1024, base_chs * ch_mult[-1])
        assert len(resolution) == n_head, "num of head must be equal to len(resolution)"

        self.fe = nn.ModuleList()
        for n in range(n_head):
            layers = []
            in_chs = base_chs * ch_mult[n]
            size = resolution[n]
            for _ in range(num_cbr):
                stride = 2 if size % 2 == 0 else 1
                chs = in_chs * 2 if in_chs * 2 <= out_chs else in_chs
                layers.append(
                    self._make_cbr(
                        in_chs, chs, stride, norm, act)
                )
                size /= 2
                in_chs *= 2 if in_chs != out_chs else 1

            layers.append(
                SEBlock(out_chs)
            )
            self.fe.append(
                nn.Sequential(*layers)
            )


        self.mlp = nn.Sequential(
            nn.Linear(out_chs * n_head, base_chs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_chs, num_classes)
        )

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor
    ):
        """
        default setting,
        Want more input argument?
        --Create more DetectHead (Default:3)
        :param x1: an [B x 128 x 112 x 112] of Tensor
        :param x2: an [B X 256 x 28 x 28] of Tensor
        :param x3: an [B X 512 X 7 X 7] of Tensor
        :return: a 2-D [B, NUM_CLASSES] of Tensor
        """
        assert len(self.fe) == self.n_head, "len of feature extracor must correspond to num head"
     
        x1 = self.fe[0](x1)
        x2 = self.fe[1](x2)
        x3 = self.fe[2](x3)

        g1 = torch.flatten(torch.mean(x1, dim=[2, 3]), 1)
        g2 = torch.flatten(torch.mean(x2, dim=[2, 3]), 1)
        g3 = torch.flatten(torch.mean(x3, dim=[2, 3]), 1)

        h = torch.cat([g1, g2, g3], dim=1)
        out = self.mlp(h)
        return out

    def _make_cbr(
            self,
            in_chs,
            out_chs,
            stride,
            norm: Optional[str],
            act: Optional[str]
    ):
        return nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 3, stride, padding=1),
            get_norm(norm, out_chs),
            get_activation(act)
        )

# ======================================================================================================================
class CRCnet(nn.Module):
    def __init__(
            self,
            # Unet argument
            image_size,
            init_ch,
            unet_base_chs,
            out_chs,
            unet_ch_mult,
            low_in_chs,
            high_in_chs,
            attn_resolution,
            num_res_block,
            num_attn_heads,
            num_attn_head_chs,
            num_heads_upsample,
            conv_resample,
            use_fp16,
            use_scale_shift_norm,
            use_resblock_updown,
            use_new_attn_order,
            # DetectHead argument
            detect_base_chs,
            detect_ch_mult,
            num_classes,
            num_cbr,
            detect_resolution,
            num_detect_head,
            dropout,
            norm,
            act,
            **kwargs
    ):
        super().__init__()
        self.loc_extract = MobileNetV1(use_extract=True, norm='gn')
        self.glo_extract = ResNet50FiLM(num_classes=num_classes, use_extract=True)
        
        self.backbone = Unet(
            image_size,
            num_classes,
            init_ch,
            unet_base_chs,
            out_chs,
            low_in_chs,
            high_in_chs,
            num_res_block,
            attn_resolution,
            dropout,
            unet_ch_mult,
            num_attn_heads,
            num_attn_head_chs,
            num_heads_upsample,
            # norm
            # act
            conv_resample=conv_resample,
            use_fp16=use_fp16,
            use_scale_shift_norm=use_scale_shift_norm,
            use_resblock_updown=use_resblock_updown,
            use_new_attn_order=use_new_attn_order
        )

        self.output = DetectHead(
            detect_base_chs,
            detect_ch_mult,
            detect_resolution,
            num_detect_head,
            num_classes,
            num_cbr,
            dropout,
            norm,
            act
        )

    def forward(self,
                x: torch.Tensor,
                t_idx: torch.Tensor,
                rois: torch.Tensor):
        loc_feats = self.loc_extract(rois)
        glo_feats = self.glo_extract(x, t_idx)
        
        pred, latent_feats = self.backbone(x, t_idx, loc_feats, glo_feats)
        
        x1, x2, x3 = latent_feats
        
        logits = self.output(x3, x2, x1)
        return logits, pred

if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # x = torch.randn(size=[4, 1, 224, 224], dtype=torch.float32, device=device)
    # low_semantic = torch.randn(size=[4, 256, 56, 56], dtype=torch.float32, device=device)
    # high_semantic = torch.randn(size=[4, 2048, 1, 1], dtype=torch.float32, device=device)

    # t_idx = torch.randint(0, 1_000, size=[x.shape[0], ])
    # model = CRCnet(
    #    image_size=224,
    #     init_ch=1,
    #     unet_base_chs=128,
    #     out_chs=1,
    #     unet_ch_mult=[1, 1, 2, 2, 4, 4],
    #     low_in_chs=256,
    #     high_in_chs=2048,
    #     attn_resolution=[],
    #     num_res_block=2,
    #     num_attn_heads=8,
    #     num_attn_head_chs=-1,
    #     num_heads_upsample=-1,
    #     conv_resample=True,
    #     use_fp16=False,
    #     use_scale_shift_norm=True,
    #     use_resblock_updown=True,
    #     use_new_attn_order=False,
    #     detect_base_chs=128,
    #     detect_ch_mult=[1, 2, 4],
    #     num_classes=3,
    #     num_cbr=5,
    #     detect_resolution=[112, 28, 7],
    #     num_detect_head=3,
    #     dropout=0.0,
    #     norm='bn',
    #     act='relu'
    # )
    # out, _ = model(x, t_idx, low_semantic, high_semantic)
    # print(out.shape)
    # print(model)
    pass
