import math
import os.path
import torch.optim.lr_scheduler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from Models.diffusion import get_named_beta_schedule, GaussianDiffusion
from Models.losses import JointLoss, cross_entropy_loss
from Utils.tools import *
from Models import (MobileNetV1,
                    ResNet50FiLM,
                    ResNet50,
                    CRCNet,
                    VGG,
                    EfficientNet)

def get_optimizer(name,
                  model_params,
                  lr=1e-3,
                  weight_decay=0.0,
                  momentum=0.9,
                  **kwargs):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(params=model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(params=model_params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(params=model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {name}")

def get_lr_scheduler(name,
                     optimizer,
                     **kwargs):
    name = name.lower()
    if name == 'none':
        return None
    elif name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=kwargs.get('lr_step', 30),
                                               gamma=kwargs.get('lr_gamma', 0.1))
    elif name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                    milestones=kwargs.get('lr_milestones', [30, 60, 90]),
                                                    gamma=kwargs.get('lr_gamma', 0.1))
    elif name == 'exponentiallr':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=kwargs.get('lr_gamma', 0.95))
    elif name == 'cosineannealinglr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                          T_max=kwargs.get('lr_T_max', 100))
    elif name == 'warmup_cosineannealing':
        def make_lambda_warmup_cosine_scheduler(optimizer,
                                                Ts,
                                                warmup_T=0,
                                                warmup_start_lr=0.0,
                                                min_lr=0.0):
            if Ts <= 0:
                raise ValueError("total_iters must be > 0")

            lr_lambdas = []
            for pg in optimizer.param_groups:
                base_lr = float(pg.get('lr', 0.0))
                if base_lr <= 0.0:
                    min_ratio = 0.0
                    warmup_start_ratio = 0.0
                else:
                    min_ratio = float(min_lr) / base_lr
                    warmup_start_ratio = float(warmup_start_lr) / base_lr

                def _make_lambda(total_iters=Ts,
                                 warmup_iters=warmup_T,
                                 min_ratio=min_ratio,
                                 warmup_start_ratio=warmup_start_ratio):
                    def lr_lambda(step: int):
                        step = int(step)
                        if total_iters <= warmup_iters and warmup_iters > 0:
                            if step >= total_iters:
                                return 1.0

                            alpha = step / float(max(1, total_iters))
                            return warmup_start_ratio + alpha * (1.0 - warmup_start_ratio)

                        # --- normal case ---
                        # warmup phase
                        if warmup_iters > 0 and step < warmup_iters:
                            alpha = step / float(max(1, warmup_iters))
                            return warmup_start_ratio + alpha * (1.0 - warmup_start_ratio)

                        # beyond total_iters -> hold at min_ratio
                        if step >= total_iters:
                            return min_ratio

                        # cosine decay phase (from base_lr -> min_lr)
                        decay_total = max(1, total_iters - warmup_iters)
                        decay_iter = max(0, step - warmup_iters)
                        t = float(decay_iter) / float(decay_total)  # in [0,1]
                        cos_out = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
                        return min_ratio + (1.0 - min_ratio) * cos_out

                    return lr_lambda

                lr_lambdas.append(_make_lambda())

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
            return scheduler

        return make_lambda_warmup_cosine_scheduler(optimizer=optimizer,
                                                   Ts=kwargs.get('Ts', 800_000),
                                                   warmup_T=kwargs.get('warmup_T', 8_000),
                                                   warmup_start_lr=kwargs.get('warmup_start_lr', 1e-3),
                                                   min_lr=kwargs.get('min_lr', 0.0))

def get_timer(name):
    if name:
        return Timer()

def get_writer(name,
               log_dir):
    if not os.path.isdir(log_dir):
        raise NotADirectoryError(f'{log_dir} is not a directory')
    if name:
        return SummaryWriter(log_dir=log_dir)

def get_criterion(name,
                  **kwargs):
    name = name.lower()
    if name == 'cross_entropy':
        return cross_entropy_loss(
            kwargs.get('proportions'), kwargs.get('device', 'cuda')
        )
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'jointloss':
        return JointLoss(
            kwargs.get('coef1', 0.8),
            kwargs.get('coef2', 0.2),
            kwargs.get('proportions'),
            kwargs.get('device', 'cuda')
        )
    else:
        raise ValueError(f"Unknown Module {name}")

def get_pipeline(name,
                 **kwargs):
    name = name.lower()
    if name == 'basic':
        return transforms.Compose([
            transforms.Resize(kwargs.get('image_size', 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ]), transforms.Compose([
            transforms.Resize(kwargs.get('image_size', 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ])
    elif name == 'strong':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(kwargs.get('image_size', 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ]), transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(kwargs.get('image_size', 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ])
    elif name == 'none':
        return None

def get_dataset(name,
                **kwargs):
    name = name.lower()
    if name == 'roi':
        image_path = kwargs.get('image_path')
        annotation_path = kwargs.get('annotation_path')
        if os.path.isdir(image_path) and os.path.isfile(annotation_path):
            return RoIDataset(image_path=image_path,
                              annotation_path=annotation_path)
    elif name == 'raw':
        image_path = kwargs.get('image_path')
        annotation_path = kwargs.get('annotation_path')
        if os.path.isdir(image_path) and os.path.isfile(annotation_path):
            return RawDataset(image_path=image_path,
                              annotation_path=annotation_path)

    elif name == 'mri':
        image_path = kwargs.get('image_path')
        annotation_path = kwargs.get('annotation_path')
        if os.path.isdir(image_path) and os.path.isfile(annotation_path):
            return MRIDataset(image_path=image_path,
                              annotation_path=annotation_path)

def get_model(name,
              **kwargs):
    name = name.lower()
    if name == 'mobilenet_v1':
        model = MobileNetV1.MobileNetV1(
            init_chs=kwargs.get('init_ch', 1),
            num_classes=kwargs.get('num_classes', 3),
            norm=kwargs.get('norm', 'bn'),
            act=kwargs.get('act', 'relu'),
            dropout=kwargs.get('dropout', 0.1)
        )
        return model

    elif name == 'resnet50':
        model = ResNet50.ResNet50(
            init_chs=kwargs.get('init_chs', 1),
            num_classes=kwargs.get('num_classes', 3),
            norm=kwargs.get('norm', 'bn'),
            act=kwargs.get('act', 'relu'),
            dropout=kwargs.get('dropout', 0.1)
        )
        return model

    elif name == 'resnet50film':
        return ResNet50FiLM.ResNet50FiLM(num_classes=kwargs.get('num_classes', 3))

    elif name == 'vgg16':
        return VGG.vgg16_bn(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'vgg19':
        return VGG.vgg19_bn(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'vgg13':
        return VGG.vgg13_bn(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'vgg11':
        return VGG.vgg11_bn(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b0':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b1':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b2':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b3':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b4':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b5':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b6':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))

    elif name == 'efficientnet_b7':
        return EfficientNet.efficientnet_factory(in_channels=kwargs.get('init_chs', 1), num_classes=kwargs.get('num_classes', 3))
    
    elif name == 'crcnet':
        return CRCNet.CRCnet(
            image_size=kwargs.get('image_size', 224),
            init_ch=kwargs.get('init_ch', 1),
            unet_base_chs=kwargs.get('unet_base_chs', 128),
            out_chs=kwargs.get('out_chs', 1),
            unet_ch_mult=kwargs.get('unet_ch_mult', [1, 1, 2, 2, 4, 4]),
            low_in_chs=kwargs.get('low_in_chs', 256),
            high_in_chs=kwargs.get('high_in_chs', 2048),
            attn_resolution=kwargs.get('attn_resolution', []),
            num_res_block=kwargs.get('num_res_block', 2),
            num_attn_heads=kwargs.get('num_attn_heads', 8),
            num_attn_head_chs=kwargs.get('num_attn_head_chs', -1),
            num_heads_upsample=kwargs.get('num_heads_upsample', -1),
            conv_resample=kwargs.get('conv_resample', True),
            use_fp16=kwargs.get('use_fp16', False),
            use_scale_shift_norm=kwargs.get('use_scale_shift_norm', True),
            use_resblock_updown=kwargs.get('use_resblock_updown', True),
            use_new_attn_order=kwargs.get('use_new_attn_order', False),
            detect_base_chs=kwargs.get('detect_base_chs', 128),
            detect_ch_mult=kwargs.get('detect_ch_mult', [1, 2, 4]),
            num_classes=kwargs.get('num_classes', 3),
            num_cbr=kwargs.get('num_cbr', 5),
            detect_resolution=kwargs.get('detect_resolution', [112, 28, 7]),
            num_detect_head=kwargs.get('num_detect_head', 3),
            dropout=kwargs.get('dropout', 0.1),
            norm=kwargs.get('norm', 'bn'),
            act=kwargs.get('act', 'relu')
        )

def get_ema_model(name,
                  model):
    if name:
        return init_ema_model(model)

def get_noise_scheduler(name,
                        **kwargs):
    name = name.lower()
    if name == 'linear':
        betas = get_named_beta_schedule(name, kwargs.get('num_timesteps', 1_000))
        return GaussianDiffusion(betas, kwargs.get('use_rescale_timesteps', False))
    elif name == 'cosine':
        betas = get_named_beta_schedule(name, kwargs.get('num_timesteps', 1_000))
        return GaussianDiffusion(betas, kwargs.get('use_rescale_timesteps', False))
