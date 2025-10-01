import os
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from Utils.tools import is_image_pth, PlotBoard

class Logger:
    def __init__(self, log_dir="logs", log_name="train.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_name)

        logging.basicConfig(
            filename=self.log_path,
            filemode="w",
            level=logging.INFO,
            format="%(message)s"
        )
        self.logger = logging.getLogger()


        self.history = {
            "train_step": [],
            "train_loss": [],
            "val_step": [],
            "val_loss": [],
            "val_metrics": {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
                "confusion_matrix": []
            },
            "lr": []
        }

    def log_base_info(self, model_name, hyperparams, dataset, device):
        self.logger.info(f"Experiment: {model_name}")
        self.logger.info(f"Hyperparams: {json.dumps(hyperparams)}")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Device: {device}")

    def log_user_info(self, *args):
        msg = ""
        for item in args:
            msg += item
        self.logger.info(msg)

    def log(
            self, train_step=None, train_loss=None, val_step=None, val_metrics=None, lr=None, time=None
    ):
        msg = ""
        if train_step is not None and val_step is not None:
            assert train_step == val_step, "when you try to integrate train_step and val_step, please make them equal"
        step = train_step or val_step
        if train_loss is not None:
            msg += f"Step [{step}] | Train Loss: {train_loss:.6f}\n"
        if val_metrics is not None:
            for tag, value in val_metrics.items():
                msg += f"Step [{step}] | {tag}: {_format_metric(value)}\n"
        if lr is not None:
            msg += f"Step [{step}] | LR: {lr:.6f}\n"
        if time is not None:
            msg += f"Step [{step}] | Time: {time}s\n"
        self.logger.info(msg)

        if train_step is not None:
            self.history["train_step"].append(step)
        if train_loss is not None:
            self.history["train_loss"].append(train_loss)
        if val_step is not None:
            self.history["val_step"].append(step)
        if val_metrics is not None and val_metrics.get('val_loss') is not None:
            self.history["val_loss"].append(val_metrics["val_loss"])
        if lr is not None:
            self.history["lr"].append(lr) 
        for key, value in self.history["val_metrics"].items():
            if val_metrics is not None and val_metrics[key] is not None: 
                value.append(val_metrics[key])


    def save_history(self, path=None):
        path = path or self.log_path.replace(".log", "_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)


class LogVisualizer:
    def __init__(self, history, figsize):
        self.history = history
        self.plotboard = PlotBoard(figsize)

    def plot_loss(self, loss_type, save_pth=None):
        assert is_image_pth(save_pth)
        os.makedirs(save_pth, exist_ok=True)
        if loss_type == 'train':
            x = _is_np(self.history["train_step"])
            y = _is_np(self.history["train_loss"])
            self.plotboard.plot(x, y, 'step', 'loss')
            plt.savefig(save_pth)

        elif loss_type == 'val':
            x = _is_np(self.history["val_step"])
            y = _is_np(self.history["val_loss"])
            self.plotboard.plot(x, y, 'step', 'loss')
            plt.savefig(save_pth)

        else:
            raise ValueError(f"Unknown loss type {loss_type}")

    def plot_metrics(self, save_pth=None):
        assert is_image_pth(save_pth)
        os.makedirs(save_pth, exist_ok=True)

        x = _is_np(self.history["val_step"])
        y = []
        legend = []
        for key, value in self.history["val_metrics"].items():
            if key != "confusion_matrix":
                legend.append(key)
                y.append(_is_np(value))
        self.plotboard.plot(x, y, 'step', 'metrics', legend)
        plt.savefig(save_pth)

def _is_np(x):
    if not type(x) == np.ndarray:
        return np.array(x)

def _format_metric(value):
    if value is None:
        return "None"
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() == 1:
                return f"{float(value.item()):.6f}"
            return str(value.tolist())
    except Exception:
        pass
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)
