import copy
import time
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from CRCDataset import *

class Timer:
    """
    Timer with extra function
    """
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        return formatted_time

    def stop(self):
        local_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        self.times.append(time.time() - self.tik)
        return self.times[-1], formatted_time

    def avg(self):
        return sum(self.times) / len(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """
    Accumulator
    add different metrics and return a dict
    e.g.
    val_metrics(class 'Accumulator') = {
        'val_loss': [1.0, 0.9, 0.8, 0.7 ...],
        'accuracy': [0.3, 0.4, 0.5, 0.6 ...]
    }
    """
    def __init__(self,
                 *args):
        """

        :param num: the number of evaluation indicators
        """
        self.metrics = {
            name: [] for name in args
        }

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].append(value)

    def reset(self):
        for name in self.metrics:
            self.metrics[name] = []

    def mean_return(self):
        return {
            name: sum(value) / len(value) if value else None
            for name, value in self.metrics.items()
        }

    def __getitem__(self, item):
        return self.metrics[item]

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum decrease in the monitored loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class GradCAM:
    """
    visualize the Gard
    """
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, outp):
            self.activations = outp.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients      # (1, C, h, w)
        acts  = self.activations    # (1, C, h, w)
        weights = grads.mean(dim=(2,3), keepdim=True)         # (1, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap_gray_tensor(gray_tensor, cam_map,
                                out_path='cam_gray_tensor_overlay.jpg',
                                mean=0.5, std=0.5):
    # 1) trans to numpy
    t = gray_tensor.detach().cpu()
    # squeeze to (H, W)
    if t.dim() == 4:
        t = t.squeeze(0)  # (1,H,W)
    if t.dim() == 3:
        t = t.squeeze(0)  # (H,W)
    # x = x_norm * std + mean
    gray = t * std + mean
    # trans to 0-255
    gray = (gray * 255.0).clamp(0, 255).numpy().astype(np.uint8)

    # 2) resize the shape
    if cam_map.shape != gray.shape:
        cam_map_resized = cv2.resize(cam_map, (gray.shape[1], gray.shape[0]))
    else:
        cam_map_resized = cam_map

    # 3) generate heatmap and overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_resized), cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_bgr, 0.6, heatmap, 0.4, 0)

    # 4) save and return
    cv2.imwrite(out_path, overlay)
    return overlay

class PlotBoard:
    def __init__(self,
                 figsize=(3.5, 2.5)):
        plt.rcParams['figure.figsize'] = figsize

    def set_axes(self,
                 axes,
                 xlabel,
                 ylabel,
                 xlim,
                 ylim,
                 xscale,
                 yscale,
                 legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def plot(self,
             X=None,
             Y=None,
             xlabel=None,
             ylabel=None,
             legend=None,
             xlim=None,
             ylim=None,
             xscale='linear',
             yscale='linear',
             fmts=('-', 'm--', 'g-.', 'r:'),
             figsize=(3.5, 2.5),
             axes=None):
        if legend is None:
            legend = []

        axes = axes if axes else plt.gca()

        def has_one_axis(X):
            return (hasattr(X, "ndim") and X.ndim == 1 or
                    isinstance(X, list) and not hasattr(X[0], "__len__"))

        if has_one_axis(X):
            X = [X]

        if Y is None:
            X, Y = [[]] * len(X), X
        elif has_one_axis(Y):
            Y = [Y]

        if len(X) != len(Y):
            X = X * len(Y)
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def get_fold_data(K: int,
                  train_transforms,
                  val_transforms,
                  dataset) -> list:
    """
    get K_fold data list
    :return: fold [[train_ds, val_ds] ---> fold: 1,
                   [train_ds, val_ds] ---> fold: 2,
                   [train_ds, val_ds] ---> fold: 3
                               ...                ]
    """
    fold = []
    assert hasattr(dataset, "__getitem__"), "expect input has attr of __getitem__"
    N = len(dataset)
    labels = np.array([dataset[i][1] for i in range(N)])
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), labels), 1):
        train_ds = SubDataset(dataset, train_idx, train_transforms)
        val_ds = SubDataset(dataset, val_idx, val_transforms)
        fold.append((train_ds, val_ds))
    return fold

def init_ema_model(model: torch.nn.Module):
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()
    return ema_model

@torch.no_grad()
def update_ema(model: torch.nn.Module,
               ema_model: torch.nn.Module,
               decay: float = 0.9999):
    # -------------------------------
    # Update EMA Model
    # -------------------------------
    msd = model.state_dict()
    emsd = ema_model.state_dict()
    for k, ema_v in emsd.items():
        model_v = msd[k]
        if not torch.is_floating_point(ema_v):
            continue
        model_v = model_v.to(ema_v.device).to(ema_v.dtype)
        # ema_v = ema_v * decay + model_v * (1-decay)
        ema_v.mul_(decay)
        ema_v.add_(model_v, alpha=1.0 - decay)




def is_image_pth(pth):
    if not isinstance(pth, str):
        return False
    clean = pth.split('?', 1)[0].split('#', 1)[0].lower()
    return clean.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.webp'))