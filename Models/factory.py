from typing import Optional, Tuple
import torch
from torch import nn

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_norm(name: Optional[str],
             num_channels: int,
             num_groups: int = 32,
             eps: float = 1e-5,
             affine: bool = True) -> nn.Module:
    """
    get normalize type and return it
    """
    name = name.lower()
    if name == 'bn':
        return nn.BatchNorm2d(num_channels, eps, affine=affine)
    elif name == 'gn':
        G = min(num_groups, num_channels)
        while G > 1 and (num_channels % G != 0):
            G-=1
        return nn.GroupNorm(G, num_channels, eps, affine=affine)
    elif name == 'syncbn':
        return nn.SyncBatchNorm(num_channels, eps, affine=affine)
    elif name == 'in':
        return nn.InstanceNorm2d(num_channels, eps, affine=affine)
    elif name == 'none':
        return nn.Identity()
    raise ValueError(f"Unknown norm_types: {name}")

def get_activation(name: Optional[str],
                   **kwargs):
    """
    get activation type and return it
    """
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=kwargs.get('inplace', True))
    elif name == 'relu6':
        return nn.ReLU6(inplace=kwargs.get('inplace', True))
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=kwargs.get('negative_slope', 0.01),
                            inplace=kwargs.get('inplace', True))
    elif name == 'silu':
        return nn.SiLU(inplace=kwargs.get('inplace', True))
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'none':
        return nn.Identity()
    raise ValueError(f"Unknown activation_types: {name}")

# ------------------------------
# NoiseScheduler
# ------------------------------
class NoiseScheduler:
    def __init__(self,
                 mode: str = 'linear',
                 num_timesteps: int = 1_000,
                 device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        if mode == 'linear':
            beta_start: float = 1e-4
            beta_end: float = 2e-2
            self.num_timesteps = num_timesteps
            self.betas = torch.linspace(start=beta_start, end=beta_end, steps=num_timesteps, dtype=torch.float32).to(device)
            self.alphas = (1.0 - self.betas).to(device)
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
            self.sqrt_one_minus_cumprod_alphas = torch.sqrt(1 - self.alphas_cumprod).to(device)
            self.sqrt_cumprod_alphas = torch.sqrt(self.alphas_cumprod).to(device)
        elif mode == 'cosine':
            s: float = 0.008        # 偏移量
            beta_max: float = 0.999
            t = torch.linspace(0, num_timesteps, num_timesteps+1, dtype=torch.float64, device=device)
            alphas_cumprod = torch.cos(((t / num_timesteps + s) / (1 + s)) * (torch.pi / 2)) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.num_timesteps = num_timesteps
            self.betas = torch.clamp(betas, min=0.0, max=beta_max).to(device)
            self.alphas = (1.0 - self.betas).to(device)
            self.alphas_cumprod = alphas_cumprod[1:].to(device)
            self.alphas_cumprod_prev = torch.cat([
                torch.tensor([1.0], device=device, dtype=torch.float32),
                alphas_cumprod[:-1]
            ], dim=0).to(device)
            self.sqrt_one_minus_cumprod_alphas = torch.sqrt(1 - self.alphas_cumprod).to(device)
            self.sqrt_cumprod_alphas = torch.sqrt(self.alphas_cumprod).to(device)

# ------------------------------
# Q_Sample
# ------------------------------
def q_sample(x0: torch.Tensor,
             t: torch.Tensor,
             noise_scheduler:NoiseScheduler,
             noise=None):
    """
    add noise
    :param x0: pure image
    :param t: appointed time step
    :param noise: Gaussian noise
    :return: noisy image
    """
    if noise is None:
        noise = torch.randn_like(x0)
    batch_sqrt_cumprod_alphas = noise_scheduler.sqrt_cumprod_alphas[t]
    batch_sqrt_one_minus_cumprod_alphas = noise_scheduler.sqrt_one_minus_cumprod_alphas[t]
    return batch_sqrt_cumprod_alphas[:,None,None,None]*x0 + batch_sqrt_one_minus_cumprod_alphas[:,None,None,None]*noise, noise

# ------------------------------
# P_Sample
# ------------------------------
@torch.no_grad()
def p_sample(model,
             x: torch.Tensor,
             t_idx: torch.Tensor,
             noise_scheduler: NoiseScheduler):
    betas = noise_scheduler.betas.to(x.device)
    alphas = noise_scheduler.alphas.to(x.device)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(x.device)

    t_idx = t_idx.view([-1, 1, 1, 1])
    beta_t = betas[t_idx].expand_as(x)
    alpha_t = alphas[t_idx].expand_as(x)
    alpha_bar_t = alphas_cumprod[t_idx].expand_as(x)

    eps_theta = model(x, t_idx.view(-1))

    pred = (x - (1 - alpha_bar_t).sqrt() * eps_theta) / alpha_bar_t.sqrt()

    coef1 = (1 / alpha_t.sqrt())
    coef2 = (1 - alpha_t) / ((1 - alpha_bar_t).sqrt())

    mean = coef1 * (x - coef2 * eps_theta)
    if t_idx[0][0] == 0:
        return mean
    else:
        noise = torch.randn_like(x)
        sigma_t = beta_t.sqrt()
        return mean + sigma_t * noise

# -------------------------------
# Denoising Models Implicit Model Scheduler
# -------------------------------
class DDIMScheduler:
    def __init__(self,
                 betas: torch.Tensor,
                 device: torch.device):
        assert betas.ndim == 1, "betas must be a one_dim tensor"
        self.device = device
        self.T = betas.shape[0]

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def get_timesteps(self,
                      num_inference_step: int):
        assert 1 <= num_inference_step <= self.T
        timesteps = torch.linspace(0, self.T - 1, num_inference_step, device=self.device)
        return timesteps.long().flip(0)

# -------------------------------
# Denoising Models Implicit Model Sample
# -------------------------------
class DDIMSampler:
    def __init__(self,
                 model: nn.Module,
                 scheduler: DDIMScheduler,
                 device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self,
               batch_size: int,
               shape: Tuple[int, int, int],
               num_inference_steps: int = 50,
               eta: float = 0.0,
               x_T: Optional[torch.Tensor] = None,
               verbose: bool = False,
               guidance_scale: Optional[float] = None):
        """

        :param batch_size: 生成的样本数量
        :param shape: 单张样本的形状
        :param num_inference_steps: 采样过程的步数
        :param eta: 随机程度 0=纯 DDIM 确定性采样     1~DDPM
        :param x_T: 可选的初始噪声   None则随机采样
        :param verbose: 是否打印进度
        :param guidance_scale: classifier-free guidance强度   None则关闭
        :return:
        """
        C, H, W = shape
        if x_T is None:
            x_t = torch.randn(size=[batch_size, C, H, W], device=self.device)
        else:
            x_t = x_T.to(self.device)

        timesteps = self.scheduler.get_timesteps(num_inference_steps)

        for i, t in enumerate(timesteps):
            t_idx = int(t)
            alpha_bar_t = self.scheduler.alphas_cumprod[t_idx]
            if i + 1 < len(timesteps):
                alpha_prev = self.scheduler.alphas_cumprod[int(timesteps[i + 1])]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)

            eps = self.model(x_t, torch.full(size=[batch_size, ], fill_value=t_idx,
                                             device=self.device, dtype=torch.long))

            x0_pred = (x_t - self.scheduler.sqrt_one_minus_alphas_cumprod[t_idx] * eps) \
                      / self.scheduler.sqrt_alphas_cumprod[t_idx]

            # 计算 sigma_t 与 c_t
            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_prev)
            )
            c_t = torch.sqrt(1.0 - alpha_prev - sigma_t ** 2)

            # 生成下一时刻 x_{t-1}
            if i + 1 < len(timesteps):
                noise = torch.randn_like(x_t) if eta > 0 else 0.0
                x_t = torch.sqrt(alpha_prev) * x0_pred + c_t * eps + sigma_t * noise
            else:
                x_t = x0_pred  # 最后一步直接输出 x0

            if verbose:
                print(f"[DDIM] Step {i + 1}/{len(timesteps)} | t={t_idx} | η={eta:.2f}")

        return x_t


