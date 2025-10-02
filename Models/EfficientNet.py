import torch
import torch.nn as nn
from typing import Callable

try:
    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
    )
except Exception:
    efficientnet_b0 = efficientnet_b1 = efficientnet_b2 = efficientnet_b3 = None
    efficientnet_b4 = efficientnet_b5 = efficientnet_b6 = efficientnet_b7 = None

VARIANTS = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
}


def _get_parent_and_attr(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _replace_first_conv(model: nn.Module, in_channels: int, pretrained: bool):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent, attr = _get_parent_and_attr(model, name)
            old_conv: nn.Conv2d = module
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None),
                padding_mode=old_conv.padding_mode,
            )

            if pretrained and hasattr(old_conv, "weight") and old_conv.weight is not None:
                with torch.no_grad():
                    old_w = old_conv.weight.data
                    if old_w.shape[1] == in_channels:
                        if new_conv.weight.shape == old_w.shape:
                            new_conv.weight.data.copy_(old_w)
                    elif old_w.shape[1] == 3:
                        averaged = old_w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                        if in_channels == 1:
                            new_conv.weight.data.copy_(averaged)
                        else:
                            new_conv.weight.data.copy_(averaged.repeat(1, in_channels, 1, 1))
                    else:
                        if in_channels <= old_w.shape[1]:
                            new_conv.weight.data.copy_(old_w[:, :in_channels, :, :])
                        else:
                            repeat_times = (in_channels + old_w.shape[1] - 1) // old_w.shape[1]
                            new_w = old_w.repeat(1, repeat_times, 1, 1)[:, :in_channels, :, :]
                            new_conv.weight.data.copy_(new_w)
                # bias
                if old_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.data.copy_(old_conv.bias.data)
                    
            setattr(parent, attr, new_conv)
            break 


def _replace_classifier_fc(model: nn.Module, num_classes: int):
    last_linear_name = None
    last_linear_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear_module = module
    if last_linear_name is None or last_linear_module is None:
        raise RuntimeError("Can not find Linear layer")
    parent, attr = _get_parent_and_attr(model, last_linear_name)
    new_fc = nn.Linear(last_linear_module.in_features, num_classes)
    setattr(parent, attr, new_fc)


def efficientnet_factory(
    variant: str = "b0",
    in_channels: int = 1,
    num_classes: int = 1000,
    pretrained: bool = False,
    progress: bool = True,
) -> nn.Module:
    variant = variant.lower()
    if variant not in VARIANTS or VARIANTS[variant] is None:
        raise ValueError(f"Unsupported variant: {variant}")

    constructor: Callable = VARIANTS[variant]
    try:
        model = constructor(pretrained=pretrained, progress=progress)  
    except TypeError:
        model = constructor(progress=progress)
    _replace_first_conv(model, in_channels=in_channels, pretrained=pretrained)
    _replace_classifier_fc(model, num_classes=num_classes)

    return model


if __name__ == "__main__":
    model = efficientnet_factory("b0", in_channels=1, num_classes=3, pretrained=False)
    print(model)

    x = torch.randn(2, 1, 224, 224)
    y = model(x)
    print("output shape:", y.shape)  # 
